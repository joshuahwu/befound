"""
Run a batch of data through befound.data.train_utils.prepare_batch (using the
parameters from a model config) and visualize the resulting 3D pose windows
with neuroposelib.vis.pose.grid3D.

Loads the disk-backed dataset (befound.get.get_rodent_data_disk) for the
datasets/split in the config, then -- rather than pulling one random batch,
which could easily under-represent whichever dataset has fewer windows --
deterministically samples exactly --n-samples windows from *each* dataset
listed in the config and assembles those into one combined batch. That batch
is run through prepare_batch exactly as trainer.py does (same
kinematic_tree/offsets/window_sizes/model_window arguments, pulled off the
dataset).

grid3D renders multiple grid panels from a single continuous pose array and a
list of reference frame indices into it, sharing one connectivity -- since our
samples are independent windows (not one continuous recording) and different
datasets have different connectivities/keypoint counts, this concatenates each
dataset's own selected windows end-to-end into one synthetic array and points
grid3D's frame references at each window's center, so each panel in a
dataset's grid shows exactly one of that dataset's sampled windows (subtitled
by sample index). One grid video is produced per dataset.
"""

import argparse
from pathlib import Path

import numpy as np
import torch
from neuroposelib import read, vis

from befound.get import get_rodent_data_disk, collate_h5_dataset
from befound.data.train_utils import prepare_batch

MODEL_CONFIG_PATH = (
    "/mnt/home/jwu10/working/ceph/results/foundation/multi_dataset/base/model_config.yaml"
)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=MODEL_CONFIG_PATH, help="Path to model_config.yaml")
    ap.add_argument("--split", default="train", choices=["train", "val", "test"])
    ap.add_argument("--n-samples", type=int, default=4, help="How many samples to render *per dataset*")
    ap.add_argument("--out-dir", default="./vis_prepare_data/")
    ap.add_argument("--n-frames", type=int, default=None, help="Video length in frames (default: model window)")
    ap.add_argument("--fps", type=int, default=25, help="Playback fps for the output video")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)

    config = read.config(args.config)
    model_window = config["model"]["window"]
    # Each sample only has model_window frames after prepare_batch, so the
    # arena3D window (centered on the same reference frame) can't exceed that.
    n_frames = min(args.n_frames or model_window, model_window)

    loader = get_rodent_data_disk(
        data_config=config["data"],
        train_val_test=args.split,
        shuffle=True,
        use_be=config["model"]["use_be"],
    )
    dataset = loader.dataset

    # Skeleton connectivity per dataset, for visualization.
    connectivities = {
        name: read.connectivity_config(
            f"{config['data']['data_path']}{name}/skeleton.yaml"
        )
        for name in config["data"]["datasets"]
    }

    skeleton_offsets = {
        k: v.to(args.device) for k, v in dataset.skeleton_offsets.items()
    }

    # Deterministically pick exactly args.n_samples windows from each dataset
    # (rather than relying on one random DataLoader batch to happen to cover
    # every dataset), using the dataset's own per-sample dataset-id tracking.
    rng = np.random.default_rng(args.seed)
    entry_dataset_idx = np.asarray(dataset._entry_dataset_idx)
    selected_indices = []
    for d_idx, dataset_name in enumerate(dataset.dataset_names):
        candidates = np.where(entry_dataset_idx == d_idx)[0]
        if len(candidates) == 0:
            print(f"Warning: no windows found for dataset '{dataset_name}', skipping.")
            continue
        n = min(args.n_samples, len(candidates))
        if n < args.n_samples:
            print(f"Warning: only {n} window(s) available for '{dataset_name}' (requested {args.n_samples}).")
        selected_indices.extend(rng.choice(candidates, size=n, replace=False).tolist())

    batch = [dataset[i] for i in selected_indices]
    data = collate_h5_dataset(batch)
    data = prepare_batch(
        data=data,
        dataset_names=dataset.dataset_names,
        augment_dict=config["train"]["augmentations"],
        kinematic_tree=dataset.kinematic_tree,
        default_offsets=None,  # target_pose isn't needed for visualization
        body_scale=dataset.body_scale,
        dataset_idx_to_n_keypts=dataset.dataset_idx_to_n_keypts,
        skeleton_offsets=skeleton_offsets,
        window_sizes=dataset.window_size,
        model_window=model_window,
        device=args.device,
        get_2d=config["model"]["is_2d"],
    )

    pose_key = "x2d" if config["model"]["is_2d"] else "x3d"
    pose_batch = data[pose_key].detach().cpu().numpy()  # (B, model_window, N_max, D)
    dataset_id = data["dataset_id"].cpu().numpy() if "dataset_id" in data else None

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    n_total = pose_batch.shape[0]  # exactly the samples we selected above
    n_videos = 0
    for dataset_name in dataset.dataset_names:
        d_idx = dataset.dataset_name_to_idx[dataset_name]
        if dataset_id is not None:
            sample_indices = np.where(dataset_id == d_idx)[0]
        else:
            sample_indices = np.arange(n_total)
        if len(sample_indices) == 0:
            continue

        n_kpts = dataset.dataset_idx_to_n_keypts[d_idx]
        windows = pose_batch[sample_indices, :, :n_kpts]  # (n_this, model_window, n_kpts, D)

        # Concatenate this dataset's independent windows end-to-end into one
        # synthetic sequence, then point grid3D at each window's center so
        # every panel shows exactly one original window.
        concatenated = windows.reshape(-1, n_kpts, windows.shape[-1])
        frame_refs = [k * model_window + model_window // 2 for k in range(len(sample_indices))]
        subtitles = [f"sample {int(i)}" for i in sample_indices]

        vid_name = f"{dataset_name}_grid.mp4"
        print(f"[{dataset_name}] {len(sample_indices)} sample(s), pose shape={windows.shape[1:]} -> {vid_name}")

        vis.pose.grid3D(
            concatenated,
            connectivities[dataset_name],
            frames=frame_refs,
            centered=True,
            subtitles=subtitles,
            title=dataset_name,
            N_FRAMES=n_frames,
            fps=args.fps,
            VID_NAME=vid_name,
            SAVE_ROOT=str(out_dir),
        )
        n_videos += 1

    print(f"Saved {n_videos} grid visualization(s) ({n_total} total samples) to {out_dir}")


if __name__ == "__main__":
    main()
