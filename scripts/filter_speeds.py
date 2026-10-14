"""
Compute mean segment (bone) lengths per dataset from each dataset's train
split, based on ceph/data/sdannce/code/get_seg_lengths.py.

Rather than loading each dataset's full pose array (100M+ frames for
sdannce/pairr24m) or materializing every frame referenced by train_inds at
once (also 10s of millions of frames -- np.unique(train_inds) turns out to
cover close to half of sdannce's raw frames, not a small subsample), this
uses befound.data.data.RodentDiskDataset purely to open/locate the right h5
file(s) per dataset (it already knows how to assemble sdannce/pairr24m's
a1/a2/single multi-file structure) and streams every *unique* frame index
referenced in its loaded train indices through in batches, computing segment
lengths on the GPU per batch and accumulating a running sum/count for the
mean -- so memory use stays bounded by BATCH_SIZE regardless of dataset size.

mouse_ephys is skipped -- its train/val/test split doesn't exist yet.
"""

import numpy as np
import pandas as pd
import torch
from neuroposelib import vis
from neuroposelib.io import read
from numpy.lib.stride_tricks import sliding_window_view

from befound.data.constants import DATA_FPS
from befound.data.data import RodentDiskDataset
from tqdm import trange

DATA_PATH = "/mnt/home/jwu10/working/ceph/data/"
DATASETS = [
    "pairr24m",
    "sdannce",
    "wu_iclr25",
    "virtual_rodent",
    "mouse_ephys",
]  # mouse_ephys: no split yet
BATCH_SIZE = 65536
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"  # _filter_speeds is h5-read-bound; GPU/batching/threading all measured slower, see below

THRESHOLD_DICT = {
    "pairr24m": 6,
    "sdannce": 3.5,
    "wu_iclr25": 3.5,
    "virtual_rodent": 3.5,
    "mouse_ephys": 3.5,
}

# Pose is rescaled by TARGET_BODY_SIZE / body_scale before the speed
# threshold below is applied, so that fixed threshold means roughly the same
# thing (a fraction of body length per ~0.5s) across every dataset's own
# native units, rather than being a fixed absolute displacement that's only
# meaningful for whichever dataset it was tuned against. body_scale comes
# from RodentDiskDataset(scale_body=True) (befound.data.data), itself
# sourced from befound.data.constants.BODY_SCALE -- the per-dataset mean
# 95th-percentile snout-tailbase distance computed by get_dataset_dims.py.
TARGET_BODY_SIZE = 50.0

def _scale_factor(body_scale, key):
    """TARGET_BODY_SIZE / body_scale[key] -- multiplying pose by this
    rescales `key`'s typical body length to TARGET_BODY_SIZE."""
    return TARGET_BODY_SIZE / body_scale[key]

def _filter_speeds(pose_h5, indices, dataset_name, threshold, scale_factor=1.0):
    """Return a boolean mask (n_windows,), True for windows where at least one
    ~0.5s sliding chunk has average speed above `threshold`.

    `indices` are full per-window frame-index arrays, shape
    (n_windows, window_size) -- see RodentDiskDataset._full_window_indices.
    Speed is frame-to-frame keypoint displacement (matching
    befound.data.data.RodentDiskDataset's remove_speed_outliers, L758-767),
    averaged first across keypoints (per frame-step) then across the frame
    steps inside each 0.5s chunk -- the two means commute, so this equals
    averaging over both at once. 0.5s is converted to a frame count using
    dataset_name's native fps (befound.data.constants.DATA_FPS), since
    `indices` are native-fps frame indices.

    Pose is multiplied by `scale_factor` (see _scale_factor/TARGET_BODY_SIZE)
    before speed is computed, so `threshold` means roughly the same thing
    across datasets with very different native units/body sizes.

    Returning the mask (rather than the filtered windows) lets the caller
    combine flags across agents for social datasets (sdannce, pairr24m) --
    e.g. drop a window if *either* agent is flagged.

    Processed BATCH_SIZE windows at a time so memory stays bounded regardless
    of how many windows there are -- expanding every window at once (the
    previous behavior) needs n_windows * window_size * n_keypoints * 3 floats
    live simultaneously, which for a dataset with tens of millions of windows
    (e.g. mouse_ephys) is a huge allocation. Each batch still only reads the
    unique frames it touches from disk (the same trick, just scoped to the
    batch) -- this was previously measured to be h5-read-bound rather than
    compute-bound, so batching here is purely about memory, not speed.
    """
    fps = DATA_FPS[dataset_name]
    chunk_frames = max(1, round(0.5 * fps))
    assert chunk_frames < indices.shape[-1]
    step = max(1, chunk_frames // 3)

    flagged = np.zeros(len(indices), dtype=bool)
    for start in trange(0, len(indices), BATCH_SIZE):
        end = start + BATCH_SIZE
        batch_indices = indices[start:end]

        # Load only the unique frames this batch's windows touch, then
        # scatter back -- avoids re-reading frames shared by overlapping/
        # nearby windows straight from the (huge, network-backed) h5 file.
        flat_inds = batch_indices.reshape(-1)
        unique_inds, inverse = np.unique(flat_inds, return_inverse=True)
        pose = (
            pose_h5[unique_inds][inverse]
            .astype(np.float32)
            .reshape(batch_indices.shape + pose_h5.shape[1:])
        )  # (b, window_size, n_kpts, 3)
        pose = pose * scale_factor

        diffs = np.diff(pose, axis=1)  # (b, window_size - 1, n_kpts, 3)
        frame_speed = np.sqrt((diffs**2).sum(-1))  # (b, window_size - 1, n_kpts)
        mean_frame_speed = frame_speed.mean(-1)  # (b, window_size - 1), keypoint-averaged

        # Sliding ~0.5s chunks over the frame-step axis, stepped rather than
        # every single frame (matches the stride used by an earlier torch.unfold call).
        chunk_speed = sliding_window_view(mean_frame_speed, chunk_frames, axis=1)[:, ::step, :].mean(-1)

        flagged[start:end] = (chunk_speed > threshold).any(axis=1)

    print("# windows flagged for speed-outlier filtering:", flagged.sum(), "of", len(indices))
    assert flagged.shape[0] == indices.shape[0]
    assert flagged.ndim == 1
    return flagged


N_PLOT = 9  # grid3D lays out 9 panels as 3x3
PLOT_DIR = "./flagged_pose_vids/"

def _plot_flagged_windows(dataset, h5_key, dataset_name, flags, connectivity, split):
    """Render a random selection of N_PLOT flagged windows for one h5_key as a
    grid3D video. grid3D takes one continuous pose array plus reference frame
    indices into it, so the chosen (independent) windows are concatenated
    end-to-end and each panel's reference frame is set to its window's center
    -- with N_FRAMES == window size and centered=True, each panel then shows
    exactly one whole window (same trick as vis_prepare_data.py)."""
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
        frames=np.arange(len(chosen), dtype=np.int64)*W,
        centered=False,
        subtitles=[f"window {int(i)}" for i in chosen],
        title=f"{split} {h5_key}: {len(flagged)} flagged windows (random {len(windows)})",
        N_FRAMES=W,
        fps=DATA_FPS[dataset_name],
        VID_NAME=f"{split}_{h5_key}_flagged.mp4",
        SAVE_ROOT=PLOT_DIR,
    )


def _filtered_inds_path(data_path, dataset_name, h5_key, split):
    """Mirror RodentDiskDataset's own {split}_inds*_2s.npy naming (see
    befound.data.data.RodentDiskDataset.__init__, L744-766) so the
    speed-filtered indices land in the same folder as -- and next to -- the
    file they were filtered from, tagged "_filtered" instead of overwriting
    it. a1/a2 share one path (they share one split already), matching how
    RodentDiskDataset loads the same indices file for both."""
    base = f"{data_path}{dataset_name}/"
    if dataset_name in ("pairr24m", "sdannce"):
        if h5_key.endswith("_a1") or h5_key.endswith("_a2"):
            return f"{base}{split}_inds_social_2s_filtered.npy"
        if h5_key.endswith("_single"):
            return f"{base}{split}_inds_single_2s_filtered.npy"
    if dataset_name == "wu_iclr25":
        return f"{base}parkinsons/{split}/inds_2s_filtered.npy"
    if dataset_name == "virtual_rodent":
        # h5_key is "virtual_rodent_{x_split}" ("base" -> no suffix, matching
        # {split}_inds_2s.npy; otherwise {split}_inds_{x_split}_2s.npy).
        x_split = h5_key[len("virtual_rodent_"):]
        suffix = "" if x_split == "base" else f"_{x_split}"
        return f"{base}{split}_inds{suffix}_2s_filtered.npy"
    return f"{base}{split}_inds_2s_filtered.npy"


def main():
    # stride=1 / no speed-outlier filtering: walk every window in each
    # dataset's split, matching the "use np.unique() on train_inds" intent as
    # closely as the data will allow.
    for split in ["train", "val", "test",]:
        print(f"Processing {split} split...")
        dataset = RodentDiskDataset(
            data_path=DATA_PATH,
            dataset_names=DATASETS,
            train_val_test=split,
            stride=1,
            min_interp_fps=25,
            balanced_sampling=False,
            virtual_rodent_x_splits = ["base", "xsession", "xsubject"],
            use_speed_filtered_inds= False,
            scale_body= True,
        )

        print(f"device={DEVICE}")
        for name in DATASETS:
            base = f"{DATA_PATH}{name}/"
            connectivity = read.connectivity_config(base + "skeleton.yaml")
            skeleton_config = read.config(base + "skeleton.yaml")
            offset = torch.tensor(
                skeleton_config["OFFSET"], dtype=torch.float32, device=DEVICE
            )

            # Social datasets (pairr24m, sdannce's pose_a1/pose_a2) share one
            # split -- a1 and a2 see the exact same windows (RodentDiskDataset
            # loads the same indices file for both), so combine their per-agent
            # flags with OR: a window is dropped if *either* agent has a
            # high-speed section. Both datasets' separate pose_single.h5
            # (sessions where only one agent was ever tracked) isn't part of
            # that pairing and keeps its own, independent flags.
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
                    f"[{name}] {h5_key}: {len(inds)} unique frames "
                    f"(of {pose_h5.shape[0]} total in file)"
                )

                # name is already the right body_scale key for every h5_key
                # -- including virtual_rodent, where it's the combined value
                # (not per-x-split), since `name` here is the base dataset
                # name "virtual_rodent", not the h5_key
                # ("virtual_rodent_base"/"_xsession"/"_xsubject"). sdannce's
                # CNTNAP sessions used to need their own separate scale_factor
                # here (CNTNAP animals are much larger), but no longer do --
                # see ceph/data/sdannce/code/fix_cntnap_size.py, which
                # rescaled pose_single.h5's CNTNAP sessions to match the rest
                # of sdannce's body size.
                flags_by_h5_key[h5_key] = _filter_speeds(
                    pose_h5, inds, name, THRESHOLD_DICT[name],
                    scale_factor=_scale_factor(dataset.body_scale, name),
                )

            # Combined (OR'd across agents) flags are used for actually
            # dropping windows below -- a paired a1/a2 window is removed if
            # *either* agent was too fast, since they share one split and
            # must stay aligned. But _plot_flagged_windows should only show
            # an agent's OWN flagged windows (own_flags_by_h5_key, computed
            # above, untouched here) -- otherwise a1's video would include
            # windows that were only flagged because of a2's speed, and vice
            # versa, misleadingly suggesting agent 1 itself moved too fast.
            save_flags_by_h5_key = dict(flags_by_h5_key)
            if social_keys:
                combined = np.logical_or(flags_by_h5_key[social_keys[0]], flags_by_h5_key[social_keys[1]])
                for k in social_keys:
                    save_flags_by_h5_key[k] = combined
                print(
                    f"[{name}] combined social-agent flags ({' + '.join(social_keys)}): "
                    f"{combined.sum()} windows flagged (either agent)"
                )

            # Save each h5_key's surviving (non-flagged) indices, in the same
            # on-disk format (center-only or full-window) as dataset.indices
            # itself, next to the original inds file. a1/a2 map to the same
            # output path (same combined flags, same shared split) so only
            # the first of the pair actually writes.
            written_paths = set()
            for h5_key in h5_keys:
                out_path = _filtered_inds_path(DATA_PATH, name, h5_key, split)
                own_flags = flags_by_h5_key[h5_key]
                save_flags = save_flags_by_h5_key[h5_key]

                # Plot before the a1/a2 dedupe below so each agent's h5_key
                # gets its own video, using only its own flags (not the
                # combined ones used for filtering).
                _plot_flagged_windows(dataset, h5_key, name, own_flags, connectivity, split)

                if out_path in written_paths:
                    continue
                keep = dataset.indices[h5_key][~save_flags]
                np.save(out_path, keep)
                written_paths.add(out_path)
                print(f"[{name}] {h5_key}: kept {len(keep)} of {len(save_flags)} windows -> {out_path}")

        dataset.close()

if __name__ == "__main__":
    main()
