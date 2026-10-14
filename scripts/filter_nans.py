"""
Further filter each dataset's offset-filtered window indices (produced by
filter_offsets.py) by NaN content: a window is flagged if any frame has a
NaN in any keypoint coordinate.

filter_speeds.py/filter_offsets.py both flag windows via `>`/`<` threshold
comparisons (chunk_speed > threshold, seg_len > max_threshold, etc.) --
and any comparison against NaN evaluates to False in both numpy and torch,
so a NaN-containing window never trips either threshold and silently
passes both filters undetected. This catches those explicitly.

Starts from each h5_key's offset-filtered indices (filter_offsets.py's
output), not the raw/speed-only-filtered ones, so this is a third filtering
stage layered on top of both previous ones, not a replacement for them.
"""

import os

import numpy as np
from tqdm import trange

from befound.data.data import RodentDiskDataset

DATA_PATH = "/mnt/home/jwu10/working/ceph/data/"
DATASETS = [
    "pairr24m",
    "sdannce",
    "wu_iclr25",
    "virtual_rodent",
    "mouse_ephys",
]
BATCH_SIZE = 65536
SAVE_INDS = True


def _filter_nans(pose_h5, indices):
    """Return a boolean mask (n_windows,), True for windows where at least
    one frame has a NaN in any keypoint coordinate.

    `indices` are full per-window frame-index arrays, shape
    (n_windows, window_size) -- see RodentDiskDataset._full_window_indices.

    Frames are deduped once across the *entire* `indices` array up front
    (same approach as filter_offsets._filter_offsets) -- these ~2s windows
    overlap heavily, so a frame's NaN status is checked once total here,
    not once per window it happens to appear in. BATCH_SIZE still bounds
    memory for the actual h5 reads, now scoped to unique frames instead of
    raw window-frame occurrences; aggregating each window's frames' flags
    back from that per-frame result is then pure indexing, with no further
    h5 reads or computation.
    """
    unique_inds, inverse = np.unique(indices.reshape(-1), return_inverse=True)
    print(
        f"{len(indices)} windows span {indices.size} frame "
        f"occurrences, {len(unique_inds)} of them unique"
    )

    frame_has_nan = np.zeros(len(unique_inds), dtype=bool)
    for start in trange(0, len(unique_inds), BATCH_SIZE):
        end = start + BATCH_SIZE
        pose = pose_h5[unique_inds[start:end]]  # (b, n_kpts, 3)
        frame_has_nan[start:end] = np.isnan(pose).any(axis=(1, 2))

    flagged = frame_has_nan[inverse].reshape(indices.shape).any(axis=1)

    print("# windows flagged for NaN filtering:", flagged.sum(), "of", len(indices))
    assert flagged.shape[0] == indices.shape[0]
    assert flagged.ndim == 1
    return flagged


def _offset_filtered_inds_path(data_path, dataset_name, h5_key, split):
    """Mirror filter_offsets._offset_filtered_inds_path's naming -- the
    files this script reads from (filter_offsets.py's own output)."""
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


def _nan_filtered_inds_path(data_path, dataset_name, h5_key, split):
    """Mirror filter_offsets._offset_filtered_inds_path's naming, chained
    on top of (not overwriting) the offset-filtered file -- tagged
    "_nanfree" so every filtering stage stays independently inspectable."""
    base = f"{data_path}{dataset_name}/"
    if dataset_name in ("pairr24m", "sdannce"):
        if h5_key.endswith("_a1") or h5_key.endswith("_a2"):
            return f"{base}{split}_inds_social_2s_filtered_offsets_nanfree.npy"
        if h5_key.endswith("_single"):
            return f"{base}{split}_inds_single_2s_filtered_offsets_nanfree.npy"
    if dataset_name == "wu_iclr25":
        return f"{base}parkinsons/{split}/inds_2s_filtered_offsets_nanfree.npy"
    if dataset_name == "virtual_rodent":
        x_split = h5_key[len("virtual_rodent_") :]
        suffix = "" if x_split == "base" else f"_{x_split}"
        return f"{base}{split}_inds{suffix}_2s_filtered_offsets_nanfree.npy"
    return f"{base}{split}_inds_2s_filtered_offsets_nanfree.npy"


def main():
    for split in ["train", "val", "test"]:
        print(f"Processing {split} split...")
        # use_speed_filtered_inds=False: only used here to open h5 files/
        # get dataset_to_h5_keys/_full_window_indices -- the actual indices
        # processed below are loaded directly from filter_offsets.py's own
        # output, per h5_key, not from this dataset's own .indices.
        dataset = RodentDiskDataset(
            data_path=DATA_PATH,
            dataset_names=DATASETS,
            train_val_test=split,
            stride=1,
            min_interp_fps=25,
            balanced_sampling=False,
            virtual_rodent_x_splits=["base", "xsession", "xsubject"],
            use_speed_filtered_inds=False,
            scale_body=False,
        )

        for name in DATASETS:
            h5_keys = dataset.dataset_to_h5_keys[name]
            social_keys = [k for k in h5_keys if k.endswith("_a1") or k.endswith("_a2")]

            flags_by_h5_key = {}
            indices_by_h5_key = {}
            for h5_key in h5_keys:
                in_path = _offset_filtered_inds_path(DATA_PATH, name, h5_key, split)
                if not os.path.exists(in_path):
                    print(f"[{name}] {h5_key}: no offset-filtered indices at {in_path}, skipping")
                    continue

                print(f"Processing {h5_key} for {name} dataset from {in_path}...")
                raw_indices = np.load(in_path)
                indices_by_h5_key[h5_key] = raw_indices
                full_indices = dataset._full_window_indices(raw_indices, name)

                pose_h5 = dataset.h5_datasets[h5_key]
                flags_by_h5_key[h5_key] = _filter_nans(pose_h5, full_indices)

            # Social datasets (pairr24m, sdannce's pose_a1/pose_a2) share one
            # split -- a window is dropped for both agents if either has a
            # NaN, matching filter_speeds.py/filter_offsets.py's combination.
            save_flags_by_h5_key = dict(flags_by_h5_key)
            present_social_keys = [k for k in social_keys if k in flags_by_h5_key]
            if len(present_social_keys) == 2:
                combined = np.logical_or(
                    flags_by_h5_key[present_social_keys[0]], flags_by_h5_key[present_social_keys[1]]
                )
                for k in present_social_keys:
                    save_flags_by_h5_key[k] = combined
                print(
                    f"[{name}] combined social-agent NaN flags ({' + '.join(present_social_keys)}): "
                    f"{combined.sum()} windows flagged (either agent)"
                )

            written_paths = set()
            for h5_key in flags_by_h5_key:
                out_path = _nan_filtered_inds_path(DATA_PATH, name, h5_key, split)
                save_flags = save_flags_by_h5_key[h5_key]

                if out_path in written_paths:
                    continue
                keep = indices_by_h5_key[h5_key][~save_flags]
                if SAVE_INDS:
                    np.save(out_path, keep)
                    written_paths.add(out_path)
                print(f"[{name}] {h5_key}: kept {len(keep)} of {len(save_flags)} windows -> {out_path}")

        dataset.close()


if __name__ == "__main__":
    main()
