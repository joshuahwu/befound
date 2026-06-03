

import argparse
import os
import warnings

import numpy as np
from scipy.interpolate import splrep, splev
from tqdm import tqdm



def _load(path: str) -> dict:
    """Load a .npy file saved by either np.save or pickle.dump."""
    result = np.load(path, allow_pickle=True)
    # np.save wraps the dict in a 0-d object array -> need .item()
    # pickle.dump saves the dict directly -> already a dict
    if isinstance(result, np.ndarray):
        return result.item()
    return result

def _interp_coord(t_valid: np.ndarray, y_valid: np.ndarray,
                  t_new: np.ndarray, spline_order: int = 3) -> np.ndarray:
    k = min(spline_order, len(t_valid) - 1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        tck = splrep(t_valid, y_valid, k=k, s=0)
    return splev(t_new, tck)


def upsample_sequence(keypoints: np.ndarray, factor: int = 3,
                      spline_order: int = 3) -> np.ndarray:
    """
    Upsample (T, num_mice, num_kp, 2) -> (T*factor, num_mice, num_kp, 2).
    (0, 0) keypoints are treated as missing and kept as (0, 0) in the output.
    """
    T, num_mice, num_kp, _ = keypoints.shape
    T_new  = T * factor
    t_orig = np.arange(T,     dtype=np.float64)
    t_new  = np.arange(T_new, dtype=np.float64) / factor

    result = np.zeros((T_new, num_mice, num_kp, 2), dtype=keypoints.dtype)

    for m in range(num_mice):
        for kp in range(num_kp):
            traj = keypoints[:, m, kp, :]
            missing_orig = (traj[:, 0] == 0) & (traj[:, 1] == 0)
            valid_orig   = ~missing_orig

            if valid_orig.sum() >= 2:
                t_v           = t_orig[valid_orig]
                t_new_clipped = np.clip(t_new, t_v[0], t_v[-1])

                for c in range(2):
                    y_v = traj[valid_orig, c]
                    try:
                        y_interp = _interp_coord(t_v, y_v, t_new_clipped, spline_order)
                        bad = ~np.isfinite(y_interp)
                        if bad.any():
                            y_interp[bad] = 0.0
                        result[:, m, kp, c] = y_interp
                    except Exception as exc:
                        print(f"  [warn] spline failed m={m} kp={kp} c={c}: {exc}")

            missing_new = np.repeat(missing_orig, factor)
            result[missing_new, m, kp, :] = 0.0

    return result



def process_file(input_path: str, output_path: str,
                 factor: int = 3, spline_order: int = 3) -> None:
    print(f"\n[load]  {input_path}")
    data = _load(input_path)

    # Pop sequences out of data so original 30Hz arrays are freed one-by-one
    # as we process them, keeping peak RAM at ~(new_so_far + 1_old_seq).
    sequences = data.pop("sequences")
    new_data  = dict(data)
    del data

    new_sequences = {}
    seq_ids = list(sequences.keys())

    for seq_id in tqdm(seq_ids,
                       desc=f"  upsampling {os.path.basename(input_path)}",
                       unit="seq"):
        seq_dict = sequences.pop(seq_id)   # free original seq after processing
        new_seq  = {}
        for field, value in seq_dict.items():
            if field == "keypoints":
                kp = np.asarray(value)
                assert kp.ndim == 4 and kp.shape[-1] == 2, \
                    f"Unexpected keypoints shape {kp.shape} for seq {seq_id}"
                new_seq["keypoints"] = upsample_sequence(kp, factor=factor,
                                                         spline_order=spline_order)
            else:
                new_seq[field] = value
        new_sequences[seq_id] = new_seq

    new_data["sequences"] = new_sequences

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    # Pre-flight: estimate serialized size and check available disk space.
    import shutil
    out_dir = os.path.dirname(os.path.abspath(output_path))
    free_bytes = shutil.disk_usage(out_dir).free
    # Rough estimate: sum of all upsampled keypoint arrays × 2 for np.save overhead
    est_bytes = sum(
        v["keypoints"].nbytes
        for v in new_sequences.values()
        if "keypoints" in v
    ) * 2
    if free_bytes < est_bytes:
        raise RuntimeError(
            f"Not enough disk space in {out_dir}: "
            f"{free_bytes / 2**30:.1f} GB free, "
            f"~{est_bytes / 2**30:.1f} GB needed. "
            f"Try writing to /hpc/group/tdunn/yw789/ instead."
        )
    print(f"[info]  disk free {free_bytes/2**30:.1f} GB, est. write size ~{est_bytes/2**30:.1f} GB")

    # Atomic write via .tmp → rename.
    # np.save wraps the dict in a 0-d object array, so the loader can call
    # np.load(..., allow_pickle=True).item() to recover the original dict.
    tmp_path = output_path.replace(".npy", ".tmp.npy")
    print(f"[save]  {output_path}  (writing via {os.path.basename(tmp_path)})")
    np.save(tmp_path, new_data)
    os.replace(tmp_path, output_path)
    print(f"[done]  {output_path}")


def verify(orig_path: str, upsampled_path: str, factor: int = 3) -> None:
    """
    Load each file once, but access sequence keypoints one at a time and
    del them immediately so peak RSS stays at ~1 sequence worth of data.

    Checks per sequence:
      1. Shape is (T * factor, num_mice, num_kp, 2).
      2. Every expanded-missing frame is exactly (0, 0).
    """
    print(f"\n[verify] {os.path.basename(orig_path)} …")

    # np.load with allow_pickle gives us the dict, but the numpy arrays inside
    # are memory-mapped / lazily backed until accessed — we access one at a time.
    orig = _load(orig_path)
    up   = _load(upsampled_path)

    orig_ids = set(orig["sequences"].keys())
    up_ids   = set(up["sequences"].keys())
    if orig_ids != up_ids:
        raise AssertionError(
            f"Sequence ID mismatch: "
            f"{len(orig_ids - up_ids)} missing, {len(up_ids - orig_ids)} extra"
        )

    errors = 0
    for i, sid in enumerate(orig["sequences"]):
        kp_o = np.asarray(orig["sequences"][sid]["keypoints"])
        kp_u = np.asarray(up  ["sequences"][sid]["keypoints"])

        T, num_mice, num_kp, _ = kp_o.shape
        expected = (T * factor, num_mice, num_kp, 2)

        if kp_u.shape != expected:
            print(f"  [FAIL] {sid}: shape {kp_u.shape} != {expected}")
            errors += 1
        else:
            miss     = (kp_o[..., 0] == 0) & (kp_o[..., 1] == 0)
            miss_exp = np.repeat(miss, factor, axis=0)
            if (kp_u[miss_exp] != 0).any():
                print(f"  [FAIL] {sid}: non-zero at missing keypoint positions")
                errors += 1

        del kp_o, kp_u          # <-- free immediately, keep RSS tiny

        if (i + 1) % 500 == 0:
            print(f"  … {i + 1} sequences checked")

    total = i + 1
    if errors == 0:
        print(f"[verify] all {total} sequences passed.\n")
    else:
        raise AssertionError(f"[verify] {errors}/{total} sequences FAILED.\n")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Upsample MABe22 mouse triplet .npy files from 30 Hz to 90 Hz.")
    p.add_argument("--input_dir",  required=True,
                   help="Directory containing mouse_triplet_train.npy / _test.npy")
    p.add_argument("--output_dir", required=True,
                   help="Directory where upsampled files will be written")
    p.add_argument("--factor",       type=int, default=3,
                   help="Upsampling factor (default 3 → 30 Hz to 90 Hz)")
    p.add_argument("--spline_order", type=int, default=3,
                   help="B-spline degree 1–5 (default 3 = cubic)")
    p.add_argument("--verify", action="store_true",
                   help="Run streaming sanity checks after each file is saved")
    p.add_argument("--files", nargs="+",
                   default=["mouse_triplet_train.npy", "mouse_triplet_test.npy"],
                   help="Files to process (default: both train and test)")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    for fname in args.files:
        in_path  = os.path.join(args.input_dir,  fname)
        out_path = os.path.join(args.output_dir, fname)

        if not os.path.exists(in_path):
            print(f"[skip] {in_path} not found.")
            continue

        process_file(in_path, out_path,
                     factor=args.factor,
                     spline_order=args.spline_order)

        if args.verify:
            verify(in_path, out_path, factor=args.factor)

    print("\nAll done.")


if __name__ == "__main__":
    main()


'''
python upsample_mabe22.py     --input_dir  /hpc/home/yw789/tdunn/befound_code/bams/data/mabe     --output_dir /hpc/home/yw789/tdunn/befound_code/bams/data/mabe_90hz     --verify
'''