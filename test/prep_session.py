"""
Preprocess one session for neural -> 3D pose decoding.

Outputs
  data/<session>_prep.npz : pose, local_qtn_full, offsets_full, yaw_full, root_full,
                            spikes_z, train_windows, test_windows
  data/<session>_meta.json: fps, offsets_sum, n_units, kept_units, zscore stats,
                            split bounds, reindex, labels, kinematic_tree, ...

Run from befound_code/neural_decode/.
"""
import os, json, argparse
import numpy as np
import h5py
from neuroposelib import read

from decode_common import (
    REINDEX, FPS, WINDOW, GAUSS_SIGMA_MS,
    compute_session_arrays, make_splits, gaussian_smooth_counts,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--h5", default="../virtual_rodent_dataset/duke/2022_02_16_1.h5")
    ap.add_argument("--skeleton", default="rat_skeleton.yaml")
    ap.add_argument("--out_dir", default="data")
    ap.add_argument("--fps", type=float, default=FPS)
    ap.add_argument("--window", type=int, default=WINDOW)
    ap.add_argument("--n_segments", type=int, default=10)
    ap.add_argument("--train_frac", type=float, default=0.8)
    ap.add_argument("--gap_sec", type=float, default=3.84)
    ap.add_argument("--train_stride", type=int, default=2)
    ap.add_argument("--test_stride", type=int, default=10)
    ap.add_argument("--min_rate_hz", type=float, default=0.0,
                    help="drop units below this rate (0 => keep all non-dead)")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    sk = read.config(args.skeleton)
    KT = sk["KINEMATIC_TREE"]
    OFF = np.array(sk["OFFSET"], dtype=np.float32)

    with h5py.File(args.h5, "r") as h:
        kp = h["pose/keypoints"][:]                       # (T, 3, 23)
        counts = h["ephys/spike_counts"][:]               # (T, 155) uint8
        active_units = np.array(h["ephys/spike_counts"].attrs["active_units"])
        session = str(h.attrs["session"])
        animal = str(h.attrs["animal"])

    pose = np.transpose(kp, (0, 2, 1))[:, REINDEX, :].astype(np.float32)   # (T, 23, 3)
    T = pose.shape[0]
    assert not np.isnan(pose).any(), "NaN in pose"
    print(f"session={session} animal={animal} T={T} pose={pose.shape} counts={counts.shape}")

    # splits (frame windows)
    gap = max(args.window, round(args.gap_sec * args.fps))
    train_windows, test_windows = make_splits(
        T, n_segments=args.n_segments, train_frac=args.train_frac, gap=gap,
        window=args.window, train_stride=args.train_stride, test_stride=args.test_stride,
    )
    train_frames = np.unique(train_windows.reshape(-1))
    test_frames = np.unique(test_windows.reshape(-1))
    overlap = np.intersect1d(train_frames, test_frames)
    assert overlap.size == 0, f"LEAKAGE: {overlap.size} frames in both train and test"
    print(f"gap={gap} frames ({gap/args.fps:.2f}s) | train_windows={train_windows.shape} "
          f"test_windows={test_windows.shape} | train_frames={train_frames.size} "
          f"test_frames={test_frames.size} | disjoint=OK")

    # spike rates: gaussian smooth (200ms), drop dead units, z-score (train stats)
    sigma_frames = GAUSS_SIGMA_MS / 1000.0 * args.fps
    rates = gaussian_smooth_counts(counts, sigma_frames)               # (T, 155)
    raw_rate_hz = counts.mean(0) * args.fps
    std_full = rates.std(0)
    kept = (std_full > 1e-6) & (raw_rate_hz >= args.min_rate_hz)
    kept_units = np.where(kept)[0]
    rates = rates[:, kept_units]
    print(f"units: total={counts.shape[1]} dropped={int((~kept).sum())} kept={kept_units.size} "
          f"(active_units flag={int(active_units.sum())})")

    tr_mean = rates[train_frames].mean(0)
    tr_std = rates[train_frames].std(0) + 1e-6
    spikes_z = ((rates - tr_mean) / tr_std).astype(np.float32)

    # window-independent per-frame arrays
    local_qtn_full, offsets_full, yaw_full, offsets_sum = compute_session_arrays(pose, KT, OFF)
    root_full = pose[:, 0, :].copy()
    print(f"offsets_sum(rat)={offsets_sum:.3f}")

    seg_bounds = np.linspace(0, T, args.n_segments + 1).astype(int).tolist()
    out_npz = os.path.join(args.out_dir, f"{session}_prep.npz")
    np.savez(
        out_npz,
        pose=pose,
        local_qtn_full=local_qtn_full,
        offsets_full=offsets_full,
        yaw_full=yaw_full,
        root_full=root_full,
        spikes_z=spikes_z,
        train_windows=train_windows,
        test_windows=test_windows,
        raw_counts_kept=counts[:, kept_units].astype(np.uint8),
    )
    meta = {
        "session": session, "animal": animal, "T": int(T),
        "fps": args.fps, "window": args.window,
        "offsets_sum": offsets_sum,
        "n_units": int(kept_units.size),
        "kept_units": kept_units.tolist(),
        "active_units": active_units.astype(bool).tolist(),
        "zscore_mean": tr_mean.tolist(), "zscore_std": tr_std.tolist(),
        "gap_frames": int(gap), "gap_sec": args.gap_sec,
        "n_segments": args.n_segments, "train_frac": args.train_frac,
        "seg_bounds": seg_bounds,
        "train_stride": args.train_stride, "test_stride": args.test_stride,
        "reindex": REINDEX, "labels": sk["LABELS"],
        "kinematic_tree": KT, "skeleton": os.path.abspath(args.skeleton),
        "n_train_windows": int(train_windows.shape[0]),
        "n_test_windows": int(test_windows.shape[0]),
        "h5": os.path.abspath(args.h5),
    }
    with open(os.path.join(args.out_dir, f"{session}_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"saved {out_npz} ({os.path.getsize(out_npz)/1e6:.1f} MB) and meta.json")


if __name__ == "__main__":
    main()
