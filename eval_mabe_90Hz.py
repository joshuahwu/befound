from __future__ import annotations

import os
from pathlib import Path
from typing import List

import numpy as np
import torch
from tqdm import trange
from numpy.lib.stride_tricks import sliding_window_view

from befound.get.get import get_model
from befound.params import read as params_read
from befound.data.train_utils import prepare_batch_2d_bespoke
from befound.data.constants import OFFSETS_3D_MABE22_SUM, OFFSETS_3D_MABE22_SUM_NO_TAIL

from sklearn.decomposition import PCA

# MABe order -> befound order
MABE_REINDEX = [6, 3, 0, 1, 2, 4, 5, 7, 8, 9, 10, 11]
EXCLUDE_KEYPTS = [10, 11]

# N_KEYPTS = 12
WINDOW = 51
HALF = WINDOW // 2  # 25

KEY_DICT = {
    # "f_vae/vae_2d":     [380, ["2d"]],
    # "f_vae/2d_td":      [400, ["2d", "3d"]],
    # "f_vae/vae":        [400, ["2d", "3d"]],
    # "ci_vae/ci_vae_2d": [265, ["2d"]],
    # "ci_vae/2d_td":     [400, ["2d", "3d"]],
    # "ci_vae/ci_vae":    [400, ["2d", "3d"]],

    # "ci_vae/2d_td":     [400, ["2d"]],
    "ci_vae/ci_vae_2d": [265, ["2d"]],
    "ci_vae/ci_vae":    [400, ["2d"]]

}


def load_mice_triplet(path):
    data_train = np.load(
        os.path.join(path, "mouse_triplet_train.npy"), allow_pickle=True
    ).item()
    sequence_ids_train, sequence_data_train = zip(*data_train["sequences"].items())
    keypoints_train = np.stack([d["keypoints"] for d in sequence_data_train])

    data_submission = np.load(
        os.path.join(path, "mouse_triplet_test.npy"), allow_pickle=True
    ).item()
    sequence_ids_submission, sequence_data_submission = zip(
        *data_submission["sequences"].items()
    )
    keypoints_submission = np.stack(
        [d["keypoints"] for d in sequence_data_submission]
    )

    sequence_ids = np.concatenate([sequence_ids_train, sequence_ids_submission], axis=0)
    keypoints = np.concatenate([keypoints_train, keypoints_submission], axis=0)

    split_mask = np.ones(len(sequence_ids), dtype=bool)
    split_mask[-len(sequence_ids_submission):] = False

    num_samples, sequence_length, num_mice, num_keypoints, _ = keypoints.shape
    keypoints = keypoints.transpose((0, 2, 1, 3, 4))
    keypoints = keypoints.reshape((-1, sequence_length, num_keypoints, 2))
    batch = np.repeat(np.arange(num_samples), num_mice)

    return keypoints, split_mask, batch


@torch.no_grad()
def encode_all_trajectories(
    model,
    keypoints: np.ndarray,   # (N, T, K, 2) float32, already reindexed
    offsets_sum: float,
    window: int = WINDOW,
    batch_size: int = 512,
    device: str = "cuda",
    pad_mode: str = "edge",
) -> np.ndarray:
    """
    Return (N, T, d_latent) float32 of per-frame latent means.

    One mouse trajectory is processed at a time to cap peak host RAM. Within
    each trajectory `batch_size` windows are pushed through the model at a
    time. We pass numpy windows straight into prepare_batch_2d_bespoke (path
    A); it handles numpy->torch conversion and the midfwd alignment itself.
    """
    def sliding_windows(seq: np.ndarray, window: int = WINDOW, pad_mode: str = "edge"):
        """
        seq: (T, K, D) numpy array for one mouse trajectory.

        Returns: (T, window, K, D) contiguous array where windows[t] is centered
        on frame t. Edge padding replicates the first/last frame (better than
        zero-pad here, because 0 is already the `missing keypoint` marker in
        MABe and would collide with true missing data).
        """
        assert seq.ndim == 3, f"expected (T, K, D), got {seq.shape}"
        half = window // 2
        padded = np.pad(seq, ((half, half), (0, 0), (0, 0)), mode=pad_mode)
        # view shape after sliding on axis 0: (T, K, D, window)
        view = sliding_window_view(padded, window_shape=window, axis=0)
        # move window axis to position 1: (T, window, K, D)
        return np.ascontiguousarray(np.moveaxis(view, -1, 1))

    N, T = keypoints.shape[:2]
    model.eval()

    z_all = None  # allocated lazily once we see d_latent

    for n in trange(N, desc="encode trajectories"):
        win = sliding_windows(keypoints[n], window=window, pad_mode=pad_mode)
        # win: (T, W, K, 2) float32

        seq_z = []
        for start in range(0, T, batch_size):
            end = min(start + batch_size, T)
            
            augment_dict = {
                    "2d_td": None,
                    "kpt_shuffle": None,
                    "offset_noise": None,
                    "single_ablation": None,
                }

            data = {"pose": win[start:end]}                     # numpy, path A
            data = prepare_batch_2d_bespoke(
                data=data,
                augment_dict=augment_dict,
                offsets_sum=offsets_sum,
                device=device,
            )
            mu = model.encode(data)["mu"]                       # (B, d)
            seq_z.append(mu.float().cpu().numpy())

        seq_z = np.concatenate(seq_z, axis=0)                   # (T, d)
        if z_all is None:
            z_all = np.empty((N, T, seq_z.shape[-1]), dtype=np.float32)
        z_all[n] = seq_z

    return z_all


def format_submission(
    per_mouse_emb: np.ndarray,
    sequence_ids: List[str],
    combine: str = "concat",
    n_mice: int = 3,
) -> dict:
    assert per_mouse_emb.ndim == 3, f"expected (N, T, d), got {per_mouse_emb.shape}"
    N, T, d_mouse = per_mouse_emb.shape
    assert N == len(sequence_ids), f"N={N} vs len(sequence_ids)={len(sequence_ids)}"
    assert N % n_mice == 0, f"N={N} not divisible by n_mice={n_mice}"
    n_seq = N // n_mice

    ordered_sids = [sequence_ids[i * n_mice] for i in range(n_seq)]
    for i, sid in enumerate(ordered_sids):
        block = sequence_ids[i * n_mice:(i + 1) * n_mice]
        assert all(s == sid for s in block), f"bad block {i}: {block}"

    grouped = per_mouse_emb.reshape(n_seq, n_mice, T, d_mouse)

    if combine == "concat":
        per_frame = grouped.transpose(0, 2, 1, 3).reshape(n_seq, T, n_mice * d_mouse)
        all_flat = per_frame.reshape(-1, n_mice * d_mouse)
        pca = PCA(n_components=128)
        per_frame = pca.fit_transform(all_flat).reshape(n_seq, T, 128)

    elif combine == "mean":
        per_frame = grouped.mean(axis=1)

    elif combine == "max":
        per_frame = grouped.max(axis=1)

    elif combine == "bams":
        # grouped: (n_seq, n_mice, T, d_mouse)
        embs_mean = grouped.mean(axis=1)            # (n_seq, T, d_mouse)
        embs_max  = grouped.max(axis=1)             # (n_seq, T, d_mouse)
        embs_min  = grouped.min(axis=1)             # (n_seq, T, d_mouse)

        per_frame = np.concatenate(
            [embs_mean, embs_max - embs_min], axis=-1
        )  # (n_seq, T, 2 * d_mouse)

        # normalize across all frames (保持和 concat 分支一致的 flatten 方式)
        flat = per_frame.reshape(-1, per_frame.shape[-1])   # (n_seq*T, 2*d_mouse)
        mean = flat.mean(axis=0, keepdims=True)
        std  = flat.std(axis=0, keepdims=True)
        per_frame = (per_frame - mean) / (std + 1e-8)       # (n_seq, T, 2*d_mouse)

    else:
        raise ValueError(f"Unknown combine={combine!r}")

    per_frame = per_frame[:, ::3, :]
    T = per_frame.shape[1]

    '''
    grouped = per_mouse_emb.reshape(n_seq, n_mice, T, d_mouse)

    sub_path = Path("submissions_1") / f"2dtd_submission_nseq_3_T_D.npy"
    np.save(sub_path, grouped)

    # if combine == "concat":
    per_frame = grouped.transpose(0, 2, 1, 3).reshape(n_seq, T, n_mice * d_mouse)
    
    sub_path = Path("submissions_1") / f"2dtd_submission_concat.npy"
    np.save(sub_path, per_frame)

    # elif combine == "mean":
    per_frame = grouped.mean(axis=1)

    sub_path = Path("submissions_1") / f"2dtd_submission_mean.npy"
    np.save(sub_path, per_frame)

    # elif combine == "max":
    per_frame = grouped.max(axis=1)

    sub_path = Path("submissions_1") / f"2dtd_submission_max.npy"
    np.save(sub_path, per_frame)
    '''

    d_frame = per_frame.shape[-1]
    if d_frame > 128:
        raise ValueError(f"Embedding dim {d_frame} exceeds MABe limit of 128.")

    embeddings = per_frame.reshape(n_seq * T, d_frame).astype(np.float32, copy=False)
    if not np.isfinite(embeddings).all():
        raise ValueError("Embeddings contain NaN/Inf.")

    frame_number_map = {sid: (i * T, (i + 1) * T) for i, sid in enumerate(ordered_sids)}
    return {"frame_number_map": frame_number_map, "embeddings": embeddings}


def validate_submission(submission: dict, submission_clips: dict) -> bool:
    if not isinstance(submission, dict): print("Submission should be dict"); return False
    if "frame_number_map" not in submission: print("frame_number_map missing"); return False
    if "embeddings" not in submission: print("embeddings missing"); return False
    emb = submission["embeddings"]
    if not isinstance(emb, np.ndarray) or emb.ndim != 2: print("embeddings not 2D ndarray"); return False
    if emb.shape[1] > 128: print(f"dim {emb.shape[1]} > 128"); return False
    if emb.dtype != np.float32: print(f"dtype {emb.dtype} != float32"); return False

    total = 0
    for sid in submission_clips["sequences"]:
        start, end = submission["frame_number_map"][sid]
        clip_len = submission_clips["sequences"][sid]["keypoints"].shape[0]
        if end - start != clip_len:
            print(f"length mismatch for {sid}"); return False
        total += clip_len
    if len(emb) != total: print("total length mismatch"); return False
    if not np.isfinite(emb).all(): print("NaN/Inf"); return False
    print("All checks passed")
    return True

def run_encode(
    data_path: str,
    results_path: str = "/mnt/home/jwu10/working/ceph/results/foundation/",
    out_dir: str = "submissions",
    combine: str = "concat",
    batch_size: int = 512,
    device: str = "cuda",
    encode_all: bool = False,      # also encode train set and save z_all
    model_keys: list | None = None,
):
    data_path = Path(data_path)
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    keys = list(KEY_DICT) if model_keys is None else list(model_keys)

    kp, split_mask, batch = load_mice_triplet(str(data_path))
    # kp:         (N*3, T, 12, 2)
    # split_mask: (N,)            True=train, False=submission
    # batch:      (N*3,)          video idx per mouse
    kp = kp[:, :, MABE_REINDEX, :].astype(np.float32, copy=False)

    if EXCLUDE_KEYPTS:
        # kp[:, :, EXCLUDE_KEYPTS, :] = 0
        mask = np.ones(kp.shape[2], dtype=bool)
        mask[EXCLUDE_KEYPTS] = False
        kp = kp[:, :, mask, :]
        # OFFSETS_3D_MABE22_SUM = OFFSETS_3D_MABE22_SUM_NO_TAIL

    per_mouse_split = split_mask[batch]            # (N*3,)
    sub_sel = ~per_mouse_split                     # True = submission mice
    kp_sub = kp[sub_sel]                           # (N_sub*3, T, 12, 2)
    kp_all = kp if encode_all else None            # kept only if requested

    # ordered submission sequence ids -> per-mouse list of length N_sub*3
    sub_raw = np.load(data_path / "mouse_triplet_test.npy", allow_pickle=True).item()
    sub_sids = list(sub_raw["sequences"].keys())
    per_mouse_sub_sids = [sid for sid in sub_sids for _ in range(3)]
    assert len(per_mouse_sub_sids) == kp_sub.shape[0], (
        f"{len(per_mouse_sub_sids)} ids vs {kp_sub.shape[0]} mice "
        "- sequence order from load_mice_triplet drifted from submission dict"
    )

    # MABe22 unit-skeleton scale (median segment-length sum, 12 keypoints).
    # See mabe22_offsets.py for how this is computed from the data.
    offsets_sum = float(OFFSETS_3D_MABE22_SUM)

    print(keys)

    for analysis_key in keys:

        load_epoch, _dim_keys = KEY_DICT[analysis_key]
        print(f"\n=== {analysis_key} (epoch {load_epoch}) ===")

        config = params_read.config_load_only(
            f"{results_path.rstrip('/')}/{analysis_key}/model_config.yaml"
        )

        config["out_path"] = f"{results_path.rstrip('/')}/{analysis_key}/"


        config["model"]["load_model"] = config["out_path"]
        config["model"]["start_epoch"] = load_epoch

        model = get_model(
            model_config=config["model"],
            load_model=config["model"]["load_model"],
            epoch=config["model"]["start_epoch"],
            n_keypts=18,
            device=device,
            verbose=1,
        )

        # encode submission (always) -- used for the .npy submission file
        z_sub = encode_all_trajectories(
            model=model, keypoints=kp_sub, offsets_sum=offsets_sum,
            window=WINDOW, batch_size=batch_size, device=device,
        )

        # for combine in ["bams", "concat", "mean", "max"]:
        for combine in ["bams"]:
            print(f"Formatting submission with combine={combine}...")

            submission = format_submission(
                per_mouse_emb=z_sub,
                sequence_ids=per_mouse_sub_sids,
                combine=combine,
                n_mice=3,
            )
            validate_submission(submission, sub_raw)
            sub_path = out_dir / f"{analysis_key.replace('/', '_')}_submission_{combine}.npy"
            np.save(sub_path, submission)
            print(f"[{analysis_key}] saved {sub_path} (d_frame={submission['embeddings'].shape[1]})")

        # optional: also encode train trajectories and dump the full array
        if encode_all:
            z_all = encode_all_trajectories(
                model=model, keypoints=kp_all, offsets_sum=offsets_sum,
                window=WINDOW, batch_size=batch_size, device=device,
            )
            z_path = out_dir / f"{analysis_key.replace('/', '_')}_z_all.npy"
            np.save(z_path, z_all)
            print(f"[{analysis_key}] saved {z_path} shape={z_all.shape}")
            del z_all

        del model, z_sub
        torch.cuda.empty_cache()


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--data_path", required=True,
                   help="dir containing mouse_triplet_train.npy and mouse_triplet_test.npy")
    p.add_argument("--results_path", default="/mnt/home/jwu10/working/ceph/results/foundation/")
    p.add_argument("--out_dir", default="submissions")
    p.add_argument("--combine", default="concat", choices=["concat", "mean", "max"])
    p.add_argument("--batch_size", type=int, default=512)
    p.add_argument("--device", default="cuda")
    p.add_argument("--encode_all", action="store_true")
    p.add_argument("--model_keys", nargs="*", default=None,
                   help="subset of KEY_DICT keys to run; default = all 6")
    args = p.parse_args()
    run_encode(**vars(args))

'''
python eval_mabe_90Hz.py \
  --data_path /hpc/home/yw789/tdunn/befound_code/bams/data/mabe_90hz/ \
  --results_path /hpc/group/tdunn/joshwu/foundation/ \
  --out_dir submissions \
  --combine max \
  --batch_size 512 \
  --device cuda
'''


'''
pip install -U aicrowd-cli
aicrowd login --api-key a249fbd253cbc8296afc4d647b5bc8c0

aicrowd submission create \
  --description "try1" \
  -c mabe-2022-mouse-triplets \
  -f ./submissions/ci_vae_2d_td_submission.npy

'''