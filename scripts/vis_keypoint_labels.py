"""
Plot one pose (one frame) per dataset in a labeled 3D scatter -- each
keypoint annotated with its string label from that dataset's skeleton.yaml
LABELS list, for visually sanity-checking keypoint indices (e.g. against
get_dataset_dims.py's SNOUT_TAILBASE_KEYPOINTS).
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from neuroposelib import read

from befound.data.data import RodentDiskDataset, fwd_kin_cont6d_torch
from befound.data.train_utils import get_midfwd_rotation

DATA_PATH = "/mnt/home/jwu10/working/ceph/data/"
DATASETS = ["mouse_ephys", "virtual_rodent", "pairr24m", "sdannce", "wu_iclr25"]
FIGURES_DIR = Path("./figures/")
DEVICE = "cpu"

# get_midfwd_rotation yaw-aligns every window to its own middle frame's
# forward direction -- empirically (checked against sdannce's L/R-labeled
# keypoints) that canonicalization always puts forward along +x, up along
# +z, and the animal's own right side along -y, regardless of dataset (the
# canonicalization itself is dataset-agnostic). A camera at elev=0, azim=-90
# sits on the -y side looking toward +y, i.e. standing on the animal's right
# side looking at it -- a level view of its right flank.
RIGHT_VIEW_ELEV, RIGHT_VIEW_AZIM = 0, -90

# Datasets in this set get their own full-width row, taller than the shared
# row the rest sit in -- mouse_ephys has far more keypoints packed close
# together than the rest, so its labels need much more room (both width AND
# height -- matplotlib's 3D axes scale to the smaller of the two to avoid
# distorting the pose, so width alone doesn't make it render bigger) to stay
# legible.
BIG_PANEL_DATASETS = {"mouse_ephys"}
BIG_ROW_HEIGHT_RATIO = 2.5


def _random_valid_frame(pose_h5, rng, batch_size=10000):
    """Index of a random frame with no NaNs: starts from a random offset and
    scans forward in batches (wrapping around once) rather than loading the
    whole (potentially huge) pose array at once, then picks a random valid
    index within the first batch that has one -- so the result isn't biased
    toward always the very first valid frame."""
    n_frames = pose_h5.shape[0]
    start0 = int(rng.integers(n_frames))
    for offset in range(0, n_frames, batch_size):
        start = (start0 + offset) % n_frames
        batch = np.asarray(pose_h5[start : start + batch_size])
        valid = ~np.isnan(batch).any(axis=(1, 2))
        if valid.any():
            valid_idx = np.flatnonzero(valid)
            return start + int(rng.choice(valid_idx))
    raise ValueError("no valid (non-NaN) frame found")


def plot_labeled_pose(ax, pose, labels, connectivity, title):
    """pose : (n_kpts, 3). Draws the skeleton (connectivity.links/colors),
    scatters keypoints, and annotates each with its label string."""
    for color, (i, j) in zip(connectivity.colors, connectivity.links):
        ax.plot3D(
            [pose[i, 0], pose[j, 0]],
            [pose[i, 1], pose[j, 1]],
            [pose[i, 2], pose[j, 2]],
            c=color, lw=2,
        )
    ax.scatter(pose[:, 0], pose[:, 1], pose[:, 2], color="black", s=20, alpha=0.3)
    for idx, label in enumerate(labels):
        ax.text(pose[idx, 0], pose[idx, 1], pose[idx, 2], f"  {idx}:{label}", fontsize=6)

    ax.set_title(title, fontsize=10)
    ax.set_box_aspect(np.ptp(pose, axis=0))
    ax.view_init(elev=RIGHT_VIEW_ELEV, azim=RIGHT_VIEW_AZIM)
    ax.grid(False)
    ax.axis(False)
    for xyz_ax in [ax.xaxis, ax.yaxis, ax.zaxis]:
        xyz_ax.set_pane_color((1, 1, 1, 0))


def main():
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    kinematic_trees, skeleton_offsets = {}, {}
    for name in DATASETS:
        skeleton_config = read.config(f"{DATA_PATH}{name}/skeleton.yaml")
        kinematic_trees[name] = skeleton_config["KINEMATIC_TREE"]
        skeleton_offsets[name] = torch.tensor(skeleton_config["OFFSET"], dtype=torch.float32)

    dataset = RodentDiskDataset(
        data_path=DATA_PATH,
        dataset_names=DATASETS,
        train_val_test="train",
        stride=1,
        min_interp_fps=25,
        balanced_sampling=False,
        virtual_rodent_x_splits=["base"],
        kinematic_tree=kinematic_trees,
        skeleton_offsets=skeleton_offsets,
    )

    big_datasets = [name for name in DATASETS if name in BIG_PANEL_DATASETS]
    small_datasets = [name for name in DATASETS if name not in BIG_PANEL_DATASETS]
    n_cols = max(len(big_datasets), len(small_datasets), 1)

    fig = plt.figure(figsize=(5 * n_cols, 5 * (BIG_ROW_HEIGHT_RATIO + 1)))
    gs = fig.add_gridspec(2, n_cols, height_ratios=[BIG_ROW_HEIGHT_RATIO, 1])
    rng = np.random.default_rng()

    def _plot(name, subplot_spec):
        skeleton_config = read.config(f"{DATA_PATH}{name}/skeleton.yaml")
        connectivity = read.connectivity_config(f"{DATA_PATH}{name}/skeleton.yaml")
        labels = skeleton_config["LABELS"]

        h5_key = dataset.dataset_to_h5_keys[name][0]
        pose_h5 = dataset.h5_datasets[h5_key]
        frame_idx = _random_valid_frame(pose_h5, rng)
        raw_pose = torch.from_numpy(np.asarray(pose_h5[frame_idx])).float()  # (n_kpts, 3)
        raw_pose = raw_pose[None, None].to(DEVICE)  # (1, 1, n_kpts, 3): batch=1, W=1

        kinematic_tree = dataset.kinematic_tree[name]
        offsets = dataset.skeleton_offsets[name].to(DEVICE)
        x6d, root, seg_offsets = get_midfwd_rotation(raw_pose, kinematic_tree, offsets)
        n_kpts = raw_pose.shape[2]
        x3d = fwd_kin_cont6d_torch(
            continuous_6d=x6d.reshape(-1, n_kpts, 6),
            kinematic_tree=kinematic_tree,
            offset=seg_offsets.reshape(-1, n_kpts, 3),
            root_pos=root.reshape(-1, 3),
            do_root_R=True,
            eps=1e-8,
        )
        pose = x3d[0].detach().cpu().numpy()  # (n_kpts, 3), canonicalized: forward=+x, up=+z

        ax = fig.add_subplot(subplot_spec, projection="3d")
        plot_labeled_pose(ax, pose, labels, connectivity, f"{h5_key} (frame {frame_idx})")

    # Big datasets each get an equal share of the full-width top row; every
    # other dataset gets its own column in the shorter bottom row.
    big_col_width = n_cols / max(len(big_datasets), 1)
    for i, name in enumerate(big_datasets):
        col_start = round(i * big_col_width)
        col_end = round((i + 1) * big_col_width)
        _plot(name, gs[0, col_start:col_end])
    for i, name in enumerate(small_datasets):
        _plot(name, gs[1, i])

    dataset.close()

    fig.suptitle("One labeled pose per dataset")
    fig.tight_layout()
    out_path = FIGURES_DIR / "keypoint_labels.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
