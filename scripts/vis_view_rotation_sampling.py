from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from neuroposelib import read, vis

import befound.data.quaternion as qtn
from befound.data.constants import DATA_FPS
from befound.data.data import RodentDiskDataset, fwd_kin_cont6d_torch
from befound.data.train_utils import _sample_view_rotation, get_midfwd_rotation

DATA_PATH = "/mnt/home/jwu10/working/ceph/data/"
DATASETS = ["mouse_ephys", "virtual_rodent", "pairr24m", "sdannce", "wu_iclr25"]

FIGURES_DIR = Path("./figures/")
POLE_CONCENTRATIONS = [1, 4, 16, 64]
N_SAMPLES = 2000
DEVICE = "cpu"

# For the grid3D pose-sequence plots (plot_random_view_pose_grids): how many
# random windows per h5_key, and the pole_concentration used for their view
# rotation (1 == fully isotropic, matching sample_axes's own leftmost panel).
N_PLOT = 9  # grid3D lays out 9 panels as 3x3
POSE_PLOT_POLE_CONCENTRATION = 1
POSE_PLOT_DIR = "./random_view_pose_vids/"


def sample_axes(pole_concentration, n_samples):
    """Viewing-axis points on the unit sphere (where the canonical +z axis
    ends up pointing) for _sample_view_rotation(pole_concentration).
    qrot(quat, z_hat) recovers exactly that axis regardless of the sampler's
    independent spin about it -- spin rotates around z_hat itself, so it
    leaves z_hat unchanged before q_axis maps it to the sampled direction.
    """
    quat = _sample_view_rotation(n_samples, DEVICE, pole_concentration=pole_concentration)
    z_hat = torch.tensor([0.0, 0.0, 1.0]).expand(n_samples, 3)
    return qtn.qrot(quat, z_hat).numpy()


def plot_unit_sphere(ax):
    u = np.linspace(0, 2 * np.pi, 30)
    v = np.linspace(0, np.pi, 30)
    x = np.outer(np.cos(u), np.sin(v))
    y = np.outer(np.sin(u), np.sin(v))
    z = np.outer(np.ones_like(u), np.cos(v))
    ax.plot_wireframe(x, y, z, color="lightgray", linewidth=0.3, alpha=0.5)


def _plot_random_view_windows(dataset, h5_key, base_name, connectivity, pole_concentration, n_plot=N_PLOT):
    """Render ONE random window for h5_key, canonicalized (get_midfwd_rotation
    + fwd_kin_cont6d_torch, the same reconstruction train_utils.prepare_batch
    uses) and then repeated n_plot times under n_plot independent random view
    rotations (_sample_view_rotation) -- i.e. exactly the pose the "rotation"
    augment_dict option produces, as a grid3D video with one panel per
    sampled view of the SAME window -- for visual sanity-checking of the view
    sampling itself (see _plot_random_view_windows's older per-window variant
    for the previous one-window-per-panel behavior)."""
    n_windows = len(dataset.indices[h5_key])
    rng = np.random.default_rng()
    chosen = int(rng.choice(n_windows))

    full = dataset._full_window_indices(dataset.indices[h5_key][chosen : chosen + 1], base_name)
    pose_h5 = dataset.h5_datasets[h5_key]
    window = np.asarray(pose_h5[full[0][0] : full[0][-1] + 1])  # (W, n_kpts, 3), contiguous slice
    W, n_kpts = window.shape[:2]
    pose = torch.from_numpy(window).float().to(DEVICE).unsqueeze(0).repeat(n_plot, 1, 1, 1)

    kinematic_tree = dataset.kinematic_tree[base_name]
    skeleton_offsets = dataset.skeleton_offsets[base_name].to(DEVICE)

    x6d, root, offsets = get_midfwd_rotation(pose, kinematic_tree, skeleton_offsets)
    x3d = fwd_kin_cont6d_torch(
        continuous_6d=x6d.reshape(-1, n_kpts, 6),
        kinematic_tree=kinematic_tree,
        offset=offsets.reshape(-1, n_kpts, 3),
        root_pos=root.reshape(-1, 3),
        do_root_R=True,
        eps=1e-8,
    ).reshape(pose.shape)  # (n_plot, W, n_kpts, 3), n_plot identical canonicalized copies

    # Panel 0 is the original (identity-rotation) view, so the random ones
    # have something unrotated to compare against; the rest get independent
    # random view rotations.
    identity_quat = torch.tensor([1.0, 0.0, 0.0, 0.0], device=DEVICE).expand(1, 4)
    random_quat = _sample_view_rotation(n_plot - 1, DEVICE, pole_concentration=pole_concentration)
    quat = torch.cat([identity_quat, random_quat], dim=0)
    q_expand = quat[:, None, None, :].expand(-1, W, n_kpts, -1)
    x3d = qtn.qrot(q_expand, x3d)

    assert torch.all(x3d[:, W // 2, 0, :].abs() < 1e-5)

    pose_np = x3d.detach().cpu().numpy().reshape(-1, n_kpts, 3)

    vis.pose.grid3D(
        pose_np,
        connectivity,
        frames=np.arange(n_plot, dtype=np.int64) * W,
        centered=False,
        subtitles=["original"] + [f"view {i}" for i in range(1, n_plot)],
        title=f"{h5_key}: window {chosen} under {n_plot} random views (pole_concentration={pole_concentration})",
        N_FRAMES=W,
        fps=DATA_FPS[base_name] // 2,
        VID_NAME=f"{h5_key}_random_view.mp4",
        SAVE_ROOT=POSE_PLOT_DIR,
    )


def plot_random_view_pose_grids():
    """Load every dataset via RodentDiskDataset and render a grid3D video per
    h5_key of random windows under a random view rotation (see
    _plot_random_view_windows)."""
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
        virtual_rodent_x_splits=["base", "xsession", "xsubject"],
        kinematic_tree=kinematic_trees,
        skeleton_offsets=skeleton_offsets,
        use_speed_filtered_inds=True,
    )

    for name in DATASETS:
        connectivity = read.connectivity_config(f"{DATA_PATH}{name}/skeleton.yaml")
        for h5_key in dataset.dataset_to_h5_keys[name]:
            print(f"Plotting {h5_key}...")
            _plot_random_view_windows(
                dataset, h5_key, name, connectivity, POSE_PLOT_POLE_CONCENTRATION
            )

    dataset.close()


def main():
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(4.5 * len(POLE_CONCENTRATIONS), 4.5))
    for i, k in enumerate(POLE_CONCENTRATIONS):
        axis = sample_axes(k, N_SAMPLES)

        ax = fig.add_subplot(1, len(POLE_CONCENTRATIONS), i + 1, projection="3d")
        plot_unit_sphere(ax)
        ax.scatter(axis[:, 0], axis[:, 1], axis[:, 2], s=4, alpha=0.6, color="tab:blue")
        ax.set_title(f"pole_concentration={k}")
        ax.set_xlim(-1, 1)
        ax.set_ylim(-1, 1)
        ax.set_zlim(-1, 1)
        ax.set_box_aspect([1, 1, 1])
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")

    fig.suptitle("Sampled viewing axes by pole_concentration")
    fig.tight_layout()
    out_path = FIGURES_DIR / "view_rotation_pole_concentration.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
    plot_random_view_pose_grids()
