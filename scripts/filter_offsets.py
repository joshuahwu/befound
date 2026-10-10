"""
Further filter each dataset's speed-filtered window indices (produced by
filter_speeds.py) by segment (bone) length: a window is flagged if any frame's
computed segment length deviates by more than THRESHOLD-fold (in either
direction) from befound.data.constants.OFFSETS_3D's default length for that
segment -- catching likely tracking errors (a limb collapsed to near-zero
length, or stretched implausibly far) that a pure speed criterion wouldn't.

Starts from each h5_key's speed-filtered indices (RodentDiskDataset(...,
use_speed_filtered_inds=True)), not the raw unfiltered ones, so this is a
second filtering stage layered on top of filter_speeds.py's, not a
replacement for it.
"""

import numpy as np
import torch
from neuroposelib import vis
from neuroposelib.io import read
from tqdm import trange

from befound.data import get_segment_len_torch
from befound.data.constants import DATA_FPS, OFFSETS_3D
from befound.data.data import RodentDiskDataset

DATA_PATH = "/mnt/home/jwu10/working/ceph/data/"
DATASETS = [
    # "pairr24m",
    # "sdannce",
    # "wu_iclr25",
    # "virtual_rodent",
    "mouse_ephys",
]
BATCH_SIZE = 65536
DEVICE = "cpu"
SAVE_INDS = True

# A window is flagged if any frame's segment length is outside
# [default / THRESHOLD, default * THRESHOLD] for any "long" segment (see
# _filter_offsets -- short segments are excluded, a ratio threshold is
# unstable for them since their absolute default length is already small).
THRESHOLD_MAX = 4.0
THRESHOLD_MIN = 50


def _filter_offsets(pose_h5, indices, dataset_name, kinematic_tree, skeleton_offsets, threshold_max=THRESHOLD_MAX, threshold_min=THRESHOLD_MIN):
    """Return a boolean mask (n_windows,), True for windows where at least one
    frame has a segment length more than `threshold`-fold above or below
    befound.data.constants.OFFSETS_3D's default length for that segment.

    `indices` are full per-window frame-index arrays, shape
    (n_windows, window_size) -- see RodentDiskDataset._full_window_indices.
    window_size here is each dataset's own ~2s window (its length depends on
    the dataset's native fps), not the model's window.

    Only "long" segments (default length > 5.0, in the dataset's own raw
    units) are checked -- a short segment's (e.g. ear-to-ear) small absolute
    default length makes a ratio-based threshold dominated by tracking
    noise rather than real deviation.

    A window is also flagged if any frame has a NaN in any keypoint
    coordinate. `>`/`<` threshold comparisons against NaN are always False
    (in both numpy and torch), so a NaN-containing window would otherwise
    silently pass the threshold check above undetected.

    Frames are deduped once across the *entire* `indices` array up front
    (rather than per BATCH_SIZE chunk of windows): these ~2s windows
    overlap heavily (consecutive windows share nearly all their frames, and
    distant windows can still revisit the same frames within a session), so
    a frame's segment length is computed once total here, not once per
    window it happens to appear in. BATCH_SIZE still bounds memory for the
    actual h5 reads + segment-length computation, now scoped to unique
    frames instead of raw window-frame occurrences; aggregating each
    window's frames' flags back from that per-frame result is then pure
    indexing, with no further h5 reads or computation.
    """
    default_offsets = torch.as_tensor(OFFSETS_3D[dataset_name], dtype=torch.float32, device=DEVICE)
    long_offsets = default_offsets > 5.0
    max_threshold = default_offsets[long_offsets] * threshold_max
    min_threshold = default_offsets[long_offsets] / threshold_min

    unique_inds, inverse = np.unique(indices.reshape(-1), return_inverse=True)
    print(
        f"[{dataset_name}] {len(indices)} windows span {indices.size} frame "
        f"occurrences, {len(unique_inds)} of them unique"
    )

    frame_flagged = np.zeros(len(unique_inds), dtype=bool)
    for start in trange(0, len(unique_inds), BATCH_SIZE):
        end = start + BATCH_SIZE
        pose = pose_h5[unique_inds[start:end]].astype(np.float32)  # (b, n_kpts, 3)

        pose_t = torch.from_numpy(pose).to(DEVICE)
        # get_segment_len_torch returns each segment's (magnitude * unit
        # direction) vector, not a scalar length -- norm it to get the
        # length actually comparable to OFFSETS_3D's magnitudes.
        seg_vecs = get_segment_len_torch(pose_t, kinematic_tree, skeleton_offsets)
        seg_len = seg_vecs.norm(dim=-1)[:, long_offsets]  # (b, n_long)

        batch_flagged = (seg_len > max_threshold[None, :]) | (seg_len < min_threshold[None, :])
        batch_flagged = batch_flagged.any(dim=-1) | torch.isnan(pose_t).any(dim=(1, 2))
        frame_flagged[start:end] = batch_flagged.cpu().numpy()

    flagged = frame_flagged[inverse].reshape(indices.shape).any(axis=1)

    print("# windows flagged for offset-outlier filtering:", flagged.sum(), "of", len(indices))
    assert flagged.shape[0] == indices.shape[0]
    assert flagged.ndim == 1
    return flagged


N_PLOT = 9  # grid3D lays out 9 panels as 3x3
PLOT_DIR = "./flagged_offset_pose_vids/"


def _plot_flagged_windows(dataset, h5_key, dataset_name, flags, connectivity, split):
    """Render a random selection of N_PLOT flagged windows for one h5_key as a
    grid3D video -- see filter_speeds._plot_flagged_windows, same approach."""
    flagged = np.flatnonzero(flags)
    if len(flagged) == 0:
        print(f"[{dataset_name}] {h5_key}: no flagged windows to plot")
        return
    rng = np.random.default_rng()
    chosen = np.sort(rng.choice(flagged, size=min(N_PLOT, len(flagged)), replace=False))

    full = dataset._full_window_indices(dataset.indices[h5_key][chosen], dataset_name)
    pose_h5 = dataset.h5_datasets[h5_key]
    windows = [np.asarray(pose_h5[w[0] : w[-1] + 1]) for w in full]  # contiguous slices
    W = windows[0].shape[0]
    pose = np.concatenate(windows, axis=0)

    vis.pose.grid3D(
        pose,
        connectivity,
        frames=np.arange(len(chosen), dtype=np.int64) * W,
        centered=False,
        subtitles=[f"window {int(i)}" for i in chosen],
        title=f"{split} {h5_key}: {len(flagged)} offset-flagged windows (random {len(windows)})",
        N_FRAMES=W,
        fps=DATA_FPS[dataset_name],
        VID_NAME=f"{split}_{h5_key}_offset_flagged.mp4",
        SAVE_ROOT=PLOT_DIR,
    )


def _offset_filtered_inds_path(data_path, dataset_name, h5_key, split):
    """Mirror filter_speeds._filtered_inds_path's naming, chained on top of
    (not overwriting) the speed-filtered file -- tagged "_offsets" so both
    filtering stages stay independently inspectable."""
    base = f"{data_path}{dataset_name}/"
    if dataset_name in ("pairr24m", "sdannce"):
        if h5_key.endswith("_a1") or h5_key.endswith("_a2"):
            return f"{base}{split}_inds_social_2s_filtered_offsets.npy"
        if h5_key.endswith("_single"):
            return f"{base}{split}_inds_single_2s_filtered_offsets.npy"
    if dataset_name == "wu_iclr25":
        return f"{base}parkinsons/{split}/inds_2s_filtered_offsets.npy"
    if dataset_name == "virtual_rodent":
        x_split = h5_key[len("virtual_rodent_") :]
        suffix = "" if x_split == "base" else f"_{x_split}"
        return f"{base}{split}_inds{suffix}_2s_filtered_offsets.npy"
    return f"{base}{split}_inds_2s_filtered_offsets.npy"


def main():
    for split in ["train", "val", "test"]:
        print(f"Processing {split} split...")
        dataset = RodentDiskDataset(
            data_path=DATA_PATH,
            dataset_names=DATASETS,
            train_val_test=split,
            strides={name: 1 for name in DATASETS},
            min_interp_fps=25,
            balanced_sampling=False,
            virtual_rodent_x_splits=["base", "xsession", "xsubject"],
            use_speed_filtered_inds=True,  # start from filter_speeds.py's output
            scale_body=False,  # OFFSETS_3D is in raw units; pose read here is too
        )

        print(f"device={DEVICE}")
        for name in DATASETS:
            kinematic_tree = dataset.kinematic_tree[name]
            skeleton_offsets = dataset.skeleton_offsets[name].to(DEVICE)
            base = f"{DATA_PATH}{name}/"
            connectivity = read.connectivity_config(base + "skeleton.yaml")

            # Social datasets (pairr24m, sdannce's pose_a1/pose_a2) share one
            # split -- a window is dropped for both agents if either is
            # flagged, matching filter_speeds.py's combination.
            h5_keys = dataset.dataset_to_h5_keys[name]
            social_keys = [k for k in h5_keys if k.endswith("_a1") or k.endswith("_a2")]

            flags_by_h5_key = {}
            for h5_key in h5_keys:
                print(f"Processing {h5_key} for {name} dataset...")
                pose_h5 = dataset.h5_datasets[h5_key]
                inds = dataset._full_window_indices(
                    dataset.indices[h5_key], dataset.h5_key_to_dataset_name[h5_key]
                )

                print(
                    f"[{name}] {h5_key}: {len(inds)} speed-filtered windows "
                    f"(of {pose_h5.shape[0]} total frames in file)"
                )

                flags_by_h5_key[h5_key] = _filter_offsets(
                    pose_h5, inds, name, kinematic_tree, skeleton_offsets
                )

            save_flags_by_h5_key = dict(flags_by_h5_key)
            if social_keys:
                combined = np.logical_or(flags_by_h5_key[social_keys[0]], flags_by_h5_key[social_keys[1]])
                for k in social_keys:
                    save_flags_by_h5_key[k] = combined
                print(
                    f"[{name}] combined social-agent offset flags ({' + '.join(social_keys)}): "
                    f"{combined.sum()} windows flagged (either agent)"
                )

            written_paths = set()
            for h5_key in h5_keys:
                out_path = _offset_filtered_inds_path(DATA_PATH, name, h5_key, split)
                own_flags = flags_by_h5_key[h5_key]
                save_flags = save_flags_by_h5_key[h5_key]

                _plot_flagged_windows(dataset, h5_key, name, own_flags, connectivity, split)

                if out_path in written_paths:
                    continue
                keep = dataset.indices[h5_key][~save_flags]
                if SAVE_INDS:
                    np.save(out_path, keep)
                    written_paths.add(out_path)
                print(f"[{name}] {h5_key}: kept {len(keep)} of {len(save_flags)} windows -> {out_path}")

        dataset.close()


if __name__ == "__main__":
    main()
