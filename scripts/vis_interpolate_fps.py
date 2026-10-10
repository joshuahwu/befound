"""
Visualize befound.data.train_utils.interpolate_center_window in isolation, on
raw pose windows straight from the dataset -- no prepare_batch, no
inverse/forward-kinematics, no augmentations. This is purely about what the
fps-interpolation function itself does to raw (x, y, z) keypoint trajectories.

Loads the disk-backed dataset (befound.get.get_rodent_data_disk) for the
datasets/split in a model config, and for --n-samples window(s) per dataset
renders two videos *per dataset* (not per sample -- all samples are appended
into the same video, via one arena3D/grid3D call each, rather than one call
per sample):

  1. {dataset}_native_full.mp4 -- the raw, unprocessed pose at the dataset's
     native fps, for its *full* native window length (no interpolation at
     all -- this is what interpolate_center_window reads from). Each
     sample's window plays in turn, appended one after another in one video.
  2. {dataset}_fps_grid.mp4 -- a neuroposelib.vis.pose.grid3D comparison of
     interpolate_center_window's output at target fps = [native, 25, 65,
     100], each cropped/interpolated down to exactly model.window frames
     (the model's window size from the config). Every sample's 4-fps
     comparison is a set of panels in the same grid (subtitled "s{i}
     <fps>fps"), so the whole dataset's samples are in one video. The
     "native" panel is an exact center crop (interpolate_center_window
     reduces to that when target_fps == native_fps), so it's a useful
     reference against which to judge the other three.
"""

import argparse
from pathlib import Path

import numpy as np
import torch
from neuroposelib import read, vis

from befound.get import get_rodent_data_disk
from befound.data.constants import DATA_FPS
from befound.data.train_utils import interpolate_center_window

MODEL_CONFIG_PATH = (
    "/mnt/home/jwu10/working/ceph/results/foundation/multi_dataset/base/model_config.yaml"
)
COMPARISON_FPS = [25, 65, 100]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=MODEL_CONFIG_PATH, help="Path to model_config.yaml")
    ap.add_argument("--split", default="train", choices=["train", "val", "test"])
    ap.add_argument("--n-samples", type=int, default=4, help="How many samples to render *per dataset*")
    ap.add_argument("--out-dir", default="./vis_interpolate_fps/")
    ap.add_argument("--interp-mode", default="linear", choices=["linear", "cubic"])
    ap.add_argument("--fps", type=int, default=25, help="Playback fps for the fps-grid video")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    config = read.config(args.config)
    model_window = config["model"]["window"]

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

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    entry_dataset_idx = np.asarray(dataset._entry_dataset_idx)

    for dataset_name in dataset.dataset_names:
        d_idx = dataset.dataset_name_to_idx[dataset_name]
        candidates = np.where(entry_dataset_idx == d_idx)[0]
        if len(candidates) == 0:
            print(f"Warning: no windows found for dataset '{dataset_name}', skipping.")
            continue

        n = min(args.n_samples, len(candidates))
        if n < args.n_samples:
            print(f"Warning: only {n} window(s) available for '{dataset_name}' (requested {args.n_samples}).")
        chosen = rng.choice(candidates, size=n, replace=False)

        n_kpts = dataset.dataset_idx_to_n_keypts[d_idx]
        W_native = dataset.window_size[dataset_name]
        native_fps = DATA_FPS[dataset_name]
        target_fps_list = [native_fps] + COMPARISON_FPS

        # Gather every sample's raw window (for the native video) and its
        # fps-interpolated variants (for the grid video) across the whole
        # dataset first, so each video is produced with a single
        # arena3D/grid3D call -- passing every sample as its own entry in
        # `frames`/`subtitles` appends them one after another (native video)
        # or as more panels in the same grid (fps-grid video), instead of
        # writing a separate video file per sample.
        native_poses = []
        interp_poses = []
        subtitles = []
        for i, idx in enumerate(chosen):
            sample = dataset[int(idx)]
            # (1, W_native, n_kpts, 3): strip padding back to this dataset's
            # real window length and keypoint count.
            pose = sample["pose"][:W_native, :n_kpts].unsqueeze(0).to(args.device)
            native_poses.append(pose[0].cpu().numpy())

            target_fps = torch.tensor(target_fps_list, dtype=pose.dtype, device=args.device)
            pose_rep = pose.repeat(len(target_fps_list), 1, 1, 1)  # (4, W_native, n_kpts, 3)
            interpolated = interpolate_center_window(
                pose_rep, native_fps, target_fps, model_window, mode=args.interp_mode
            )  # (4, model_window, n_kpts, 3)
            interp_poses.append(interpolated.detach().cpu().numpy())
            subtitles += [f"s{i} native ({native_fps}fps)"] + [f"s{i} {f}fps" for f in COMPARISON_FPS]

        # 1. Raw, unprocessed pose at native fps, full native window -- every
        # sample gets its own panel in one grid3D call (grid3D, not arena3D:
        # arena3D overlays multiple `frames` as simultaneous skeletons
        # sharing one 3D view, meant for several animals in one real arena
        # at once -- wrong model for our unrelated samples from different
        # times, which grid3D instead keeps in clearly separate panels).
        native_concat = np.concatenate(native_poses, axis=0)  # (n*W_native, n_kpts, 3)
        native_frame_refs = [k * W_native + W_native // 2 for k in range(len(chosen))]
        native_subtitles = [f"s{i}" for i in range(len(chosen))]

        vid_name_native = f"{dataset_name}_native_full.mp4"
        print(
            f"[{dataset_name}] native full window x{len(chosen)}: "
            f"{native_poses[0].shape} @ {native_fps}fps -> {vid_name_native}"
        )
        vis.pose.grid3D(
            native_concat,
            connectivities[dataset_name],
            frames=native_frame_refs,
            centered=True,
            subtitles=native_subtitles,
            title=f"{dataset_name} (native, full window)",
            N_FRAMES=W_native,
            fps=native_fps,
            VID_NAME=vid_name_native,
            SAVE_ROOT=str(out_dir),
        )

        # 2. interpolate_center_window at [native, 25, 65, 100] fps, each
        # cropped/interpolated to model_window frames -- every sample's 4
        # panels are part of the same grid.
        interp_concat = np.concatenate(interp_poses, axis=0).reshape(-1, n_kpts, native_poses[0].shape[-1])
        interp_frame_refs = [k * model_window + model_window // 2 for k in range(len(chosen) * len(target_fps_list))]

        vid_name_grid = f"{dataset_name}_fps_grid.mp4"
        print(
            f"[{dataset_name}] fps grid ({args.interp_mode}) x{len(chosen)} samples, "
            f"{target_fps_list} fps -> {vid_name_grid}"
        )
        vis.pose.grid3D(
            interp_concat,
            connectivities[dataset_name],
            frames=interp_frame_refs,
            centered=True,
            subtitles=subtitles,
            title=f"{dataset_name} ({args.interp_mode} interpolation)",
            N_FRAMES=model_window,
            fps=args.fps,
            VID_NAME=vid_name_grid,
            SAVE_ROOT=str(out_dir),
        )

    print(f"Saved visualizations to {out_dir}")


if __name__ == "__main__":
    main()
