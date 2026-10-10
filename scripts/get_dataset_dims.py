from pathlib import Path

import numpy as np
import pandas as pd
import torch
import tqdm
from matplotlib import pyplot as plt
from neuroposelib import read

from befound.data.data import RodentDiskDataset

DATA_PATH = "/mnt/home/jwu10/working/ceph/data/"
FIGURES_DIR = Path("/mnt/home/jwu10/working/befound/scripts/figures/")

DATASETS = ["mouse_ephys", "virtual_rodent", "pairr24m", "sdannce", "wu_iclr25"]

# Option (in place of the old per-frame centering in session_dims_for): pass
# each dataset's 2s windows (the on-disk "_2s" split windows, same ones
# RodentDiskDataset trains on) through canonicalize_window_cartesian -- a
# pure Cartesian translate + yaw-rotate per window (keypoint 0 at the center
# frame -> origin, keypoint0->keypoint1 direction at that frame -> +x) --
# before measuring ranges, so results aren't confounded by whichever
# arbitrary direction the animal happened to be facing in the arena. See
# session_dims_for_windows.
USE_MIDFWD_ROTATION = True
BATCH_SIZE = 4096  # windows per canonicalize_window_cartesian batch
DEVICE = "cpu"#"cuda" if torch.cuda.is_available() else "cpu"

# Option, only meaningful when USE_MIDFWD_ROTATION is False (asserted in
# main()): instead of session_dims_for's per-frame max-min bounding-box range
# across all keypoints/dims, return the per-frame Euclidean distance between
# the snout and tailbase keypoints -- a single body-length-like number per
# session rather than a 3-dim range. See SNOUT_TAILBASE_KEYPOINTS for
# datasets whose skeleton has no literal snout/tailbase keypoints.
USE_SNOUT_TAILBASE_DIST = False

# (snout_idx, tailbase_idx) per dataset, resolved against each dataset's own
# skeleton.yaml LABELS list. pairr24m has no literal snout/tailbase keypoints
# at all, so its pair is HeadF (front of head) and SpineL (lower spine), its
# closest analogues.
SNOUT_TAILBASE_KEYPOINTS = {
    "mouse_ephys": (5, 7),      # Snout, TailBase
    "virtual_rodent": (2, 6),   # Snout, TailBase
    "pairr24m": (4, 7),         # HeadF, SpineL
    "sdannce": (4, 6),          # Snout, TailBase
    "wu_iclr25": (4, 5),        # Snout, Tail_base
}
SCALE_BODY_LENGTH = False

# Per-dataset session-level categorical metadata to color points by, where the
# schema is known: (path relative to DATA_PATH, column to color by, index_col
# for pd.read_csv). A dataset not listed here is plotted in a single,
# uncategorized color instead.
#
# index_col=None means "use the row position" (0, 1, 2, ... in file order) as
# the session id -- correct for wu_iclr25 and pairr24m, whose metadata.csv row
# position matches the dense 0..N-1 session ids used across ALL of a dataset's
# h5_keys (e.g. pairr24m_a1/_a2/_single each hold a disjoint subset of ids
# 0..39, see ceph/data/pairr24m/code/split_pose_single.py -- row 7 of
# metadata.csv is still session 7 regardless of which h5_key it ended up in),
# but neither file has its own explicit, h5-id-aligned id column (mouse_ephys
# and virtual_rodent do, and use that column instead). pairr24m's own
# "RecordingIndex" column is NOT usable for this -- it has gaps (1..62 over
# only 40 rows), so it doesn't match the h5's dense 0..39 session ids; row
# position does.
RATMETA_BY_DATASET = {
    "sdannce": ("sdannce/metadata.csv", "ratgroup", None),
    "virtual_rodent": ("virtual_rodent/meta.csv", "animal", 0),
    "pairr24m": ("pairr24m/metadata.csv", "Rat1", None),
    "wu_iclr25": ("wu_iclr25/parkinsons/metadata.csv", "Condition", None),
    "mouse_ephys": ("mouse_ephys/meta.csv", "Area", "session_id"),
}

# For these h5_keys, sessions matching the given ratmeta value are plotted
# separately from the rest of that h5_key rather than lumped into one shared
# plot -- useful when a subgroup's range statistics are visibly different
# from the rest of the h5_key and would otherwise dominate the shared axis
# scale, squishing every other session into one corner. Empty for now:
# sdannce_single's CNTNAP sessions used to need this (before
# ceph/data/sdannce/code/fix_cntnap_size.py rescaled them to match the rest
# of sdannce's body size), but no longer do.
SEPARATE_GROUP_BY_H5_KEY = {}

# Pooling for the dataset-level mean-of-per-session-95th-percentile summary
# (see compute_dataset_means): group label -> member group_source_keys to
# pool sessions from before averaging. A group_source_key is a h5_key, or
# (for a h5_key split by SEPARATE_GROUP_BY_H5_KEY) "{h5_key}|{value}" /
# "{h5_key}|excl_{value}". Any group_source_key not claimed by an entry here
# (e.g. mouse_ephys, wu_iclr25) automatically gets its own singleton group
# (see main()), so every h5_key/subset ends up in exactly one summary row
# unless it's also listed in a combined group here.
EXPLICIT_MEAN_GROUPS = {
    "sdannce (a1+a2+single)": ["sdannce_a1", "sdannce_a2", "sdannce_single"],
    "pairr24m (a1+a2+single)": ["pairr24m_a1", "pairr24m_a2", "pairr24m_single"],
    "virtual_rodent_base": ["virtual_rodent_base"],
    "virtual_rodent_xsession": ["virtual_rodent_xsession"],
    "virtual_rodent_xsubject": ["virtual_rodent_xsubject"],
    "virtual_rodent (combined)": [
        "virtual_rodent_base", "virtual_rodent_xsession", "virtual_rodent_xsubject",
    ],
}


def compute_dataset_means(session_dims_by_key, dim_columns, explicit_groups=EXPLICIT_MEAN_GROUPS):
    """Mean (across sessions) of each per-session 95th-percentile dim value,
    pooled per explicit_groups where given (sessions from every member
    group_source_key concatenated before averaging, so e.g. sdannce's a1/a2/
    single sessions count equally rather than averaging three already-averaged
    h5_key means), plus one singleton group per
    group_source_key not claimed by any explicit group.
    """
    covered = {key for members in explicit_groups.values() for key in members}
    groups = dict(explicit_groups)
    for key in session_dims_by_key:
        if key not in covered:
            groups[key] = [key]

    rows = []
    for label, members in groups.items():
        pooled = [session_dims_by_key[m] for m in members if m in session_dims_by_key]
        if not pooled:
            continue
        pooled = np.concatenate(pooled, axis=0)
        row = {"group": label, "n_sessions": len(pooled)}
        row.update({col: pooled[:, i].mean() for i, col in enumerate(dim_columns)})
        rows.append(row)
    return pd.DataFrame(rows)


def session_dims_for(pose_h5, ids, unique_sessions, keypoint_pair=None):
    """Per-session 95th-percentile (over frames) range (max - min) of each
    dimension, after centering every frame on keypoint 0 (so keypoint 0 sits
    at the origin).

    If keypoint_pair (snout_idx, tailbase_idx) is given (see
    SNOUT_TAILBASE_KEYPOINTS and USE_SNOUT_TAILBASE_DIST), returns the
    per-session 95th-percentile snout-to-tailbase Euclidean distance instead
    -- a single body-length-like number per session, shape (n_sessions, 1),
    rather than the 3-dim bounding-box range.
    """
    n_cols = 1 if keypoint_pair is not None else 3
    dims = np.zeros((len(unique_sessions), n_cols))
    for i, session in enumerate(tqdm.tqdm(unique_sessions)):
        session_mask = ids == session
        pose_h5_session = pose_h5[session_mask]
        if keypoint_pair is not None:
            snout_idx, tailbase_idx = keypoint_pair
            dist = np.linalg.norm(
                pose_h5_session[:, snout_idx] - pose_h5_session[:, tailbase_idx], axis=-1
            )  # (n_frames,)
            dims[i, 0] = np.percentile(dist, 95)
        else:
            pose_h5_session = pose_h5_session - pose_h5_session[:, 0:1, :]  # center on keypoint 0
            sample_ranges = pose_h5_session.max(axis=1) - pose_h5_session.min(axis=1)  # (n_frames, 3)
            dims[i] = np.percentile(sample_ranges, 95, axis=0)
    return dims


def _window_sessions(dataset, h5_key):
    """Per-window session id, aligned 1:1 with dataset.indices[h5_key] (the
    on-disk "_2s" split windows) -- a window never straddles a session
    boundary, so its center frame's id is the whole window's session."""
    ids = np.asarray(dataset.h5_ids[h5_key])
    indices = dataset.indices[h5_key]  # (n_windows,) center-only, or (n_windows, window) legacy full-window
    window_centers = indices if indices.ndim == 1 else indices[:, indices.shape[1] // 2]
    return ids[window_centers]


def session_window_counts(dataset, h5_key, unique_sessions):
    """Number of 2s windows (dataset.indices[h5_key] rows) per session, in
    unique_sessions's order -- NOT a raw frame count."""
    window_session = _window_sessions(dataset, h5_key)
    session_ids, counts = np.unique(window_session, return_counts=True)
    count_by_session = dict(zip(session_ids.tolist(), counts.tolist()))
    return np.array([count_by_session.get(int(s), 0) for s in unique_sessions], dtype=np.int64)


def _body_length_scale(dataset, h5_key, base_name, window_session):
    """Per-window divisor into dataset.body_scale, shape matching
    window_session. If h5_key is split by SEPARATE_GROUP_BY_H5_KEY (currently
    empty -- see its own comment), that subgroup mixes a population with a
    very different body size, so its windows need their own, separately-
    scaled divisor (dataset.body_scale[f"{base_name}_{separate_value}"])
    rather than sharing base_name's overall value -- every other h5_key just
    uses dataset.body_scale[base_name] uniformly (for virtual_rodent,
    base_name is "virtual_rodent" regardless of x_split, so this is already
    the combined value, not a per-split one).
    """
    default = dataset.body_scale[base_name]
    separate_value = SEPARATE_GROUP_BY_H5_KEY.get(h5_key)
    if separate_value is None:
        return np.full(len(window_session), default, dtype=np.float32)

    meta_path, ratmeta_key, index_col = RATMETA_BY_DATASET[base_name]
    meta = pd.read_csv(Path(DATA_PATH) / meta_path, index_col=index_col)
    separate_sessions = meta.index[meta[ratmeta_key] == separate_value].to_numpy()

    separate_length = dataset.body_scale[f"{base_name}_{separate_value}"]
    is_separate = np.isin(window_session, separate_sessions)
    return np.where(is_separate, separate_length, default).astype(np.float32)


def canonicalize_window_cartesian(pose):
    """Cartesian-only stand-in for befound.data.train_utils.get_midfwd_rotation
    + fwd_kin_cont6d_torch, for callers that just want the same canonicalized
    x3d (not the intermediate x6d/root/offsets, and not a kinematic-tree-
    consistent reconstruction) -- a single rigid translate + rotate per
    window, fully vectorized, with no inverse/forward-kinematics chain
    traversal. This also sidesteps get_midfwd_rotation's large-raw-coordinate
    slowdown (the IK/FK path's cross-product and normalization math can land
    on subnormal floats, which most CPUs handle via a slow microcode path --
    a rigid transform has no such intermediate terms).

    For each window: keypoint 0 at the window's own center frame is
    translated to the origin, and the whole window is rotated about z so
    that the center frame's keypoint0->keypoint1 direction points along +x
    (i.e. yaw removed) -- every other frame/keypoint just inherits that one
    per-window translation + rotation, exactly like get_midfwd_rotation's
    own root/x6d end up representing.

    Parameters
    ----------
    pose : torch.Tensor
        Shape (B, W, N_K, 3).

    Returns
    -------
    torch.Tensor
        Shape (B, W, N_K, 3), canonicalized as described above.
    """
    B, W, N_K, _ = pose.shape
    mid = W // 2

    root = pose[:, mid, 0, :]  # (B, 3): keypoint 0 at the center frame
    pose = pose - root[:, None, None, :]

    forward = pose[:, mid, 1, :] - pose[:, mid, 0, :]  # (B, 3)
    yaw = torch.atan2(forward[:, 1], forward[:, 0])  # (B,)

    # Rotate by -yaw about z so `forward`'s xy-angle becomes 0 (+x-aligned).
    cos, sin = torch.cos(yaw), torch.sin(yaw)
    R = torch.zeros((B, 3, 3), dtype=pose.dtype, device=pose.device)
    R[:, 0, 0] = cos
    R[:, 0, 1] = sin
    R[:, 1, 0] = -sin
    R[:, 1, 1] = cos
    R[:, 2, 2] = 1.0

    pose = torch.einsum("bij,bwkj->bwki", R, pose)
    assert (pose[:, mid, 0, :].norm(dim=1).abs() < 1e-6).all()
    assert (pose[:, mid, 1, 1].abs() < 1e-3).all()

    # import pdb; pdb.set_trace()
    # from neuroposelib import vis
    # connectivity = read.connectivity_config("/mnt/home/jwu10/working/ceph/data/mouse_ephys/skeleton.yaml")
    # vis.pose.grid3D(
    #     pose[[100, 2000, 4000]].cpu().detach().numpy().reshape(-1, N_K, 3),
    #     connectivity=connectivity,
    #     frames = np.arange(3,dtype=np.int32)*W,
    #     centered=False,
    #     N_FRAMES=W,
    #     fps=30,
    #     VID_NAME="midfwd_vector_test.mp4",
    #     SAVE_ROOT="./figures/",
    # )

    return pose


def session_dims_for_windows(dataset, h5_key, base_name, unique_sessions):
    """Like session_dims_for, but using the dataset's 2s windows (same
    on-disk "_2s" split windows RodentDiskDataset trains on) and
    canonicalize_window_cartesian instead of raw per-frame centering: each
    window is rigidly translated + yaw-rotated (keypoint 0 at the window's
    own center frame -> origin, keypoint0->keypoint1 direction at that frame
    -> +x) -- a pure Cartesian transform, not a kinematic-tree reconstruction
    -- and the per-dimension range (max - min over every frame and keypoint
    in the window, i.e. the window's own bounding-box extent) is measured on
    THAT pose, so a window's range isn't confounded by whichever arbitrary
    direction the animal happened to be facing in the arena. Per-session
    values are then the 95th percentile of that per-window range, over all
    of that session's windows (mirrors session_dims_for's per-frame ->
    per-session 95th-percentile aggregation, just with "window" as the base
    unit instead of "frame").

    Batched over windows (same trick as filter_speeds.py's _filter_speeds) so
    memory/compute stay bounded regardless of how many windows a dataset has
    -- unlike session_dims_for, which reads a whole session's frames at once,
    a session's own window count can run into the millions for the larger
    datasets (stride=1 sliding windows), so this batches across ALL of a
    h5_key's windows rather than looping session-by-session.
    """
    indices = dataset.indices[h5_key]  # (n_windows,) center-only, or (n_windows, window) legacy full-window
    pose_h5 = dataset.h5_datasets[h5_key]

    window_session = _window_sessions(dataset, h5_key)
    if SCALE_BODY_LENGTH:
        body_length_scale = torch.tensor(
            _body_length_scale(dataset, h5_key, base_name, window_session),
            dtype=torch.float32, device=DEVICE,
        )

    session_values = {int(s): [] for s in unique_sessions}

    for start in tqdm.trange(0, len(indices), BATCH_SIZE):
        end = start + BATCH_SIZE
        batch = indices[start:end]
        full_windows = dataset._full_window_indices(batch, base_name)  # (b, window)

        # Load only the unique frames this batch's windows touch, then
        # scatter back -- avoids re-reading frames shared by overlapping/
        # nearby windows straight from the (huge, network-backed) h5 file.
        flat_inds = full_windows.reshape(-1)
        unique_inds, inverse = np.unique(flat_inds, return_inverse=True)
        pose_np = (
            pose_h5[unique_inds][inverse]
            .astype(np.float32)
            .reshape(full_windows.shape + pose_h5.shape[1:])
        )  # (b, window, n_kpts, 3)
        pose = torch.from_numpy(pose_np).to(DEVICE)

        if SCALE_BODY_LENGTH:
            pose = pose / body_length_scale[start:end, None, None, None]

        x3d = canonicalize_window_cartesian(pose)  # (b, window, n_kpts, 3)

        window_range = x3d.amax(dim=(1, 2)) - x3d.amin(dim=(1, 2))  # (b, 3): max-min over the whole window
        window_range = window_range.detach().cpu().numpy()

        for session, r in zip(window_session[start:end], window_range):
            session_values[int(session)].append(r)

    dims = np.zeros((len(unique_sessions), 3))
    for i, session in enumerate(unique_sessions):
        values = session_values[int(session)]
        if values:
            dims[i] = np.percentile(np.stack(values), 95, axis=0)
    return dims

def plot_bar_with_dots(entries, dim_columns, title, out_path):
    """One bar-plus-dots panel per column in dim_columns, x-axis shared
    across ALL of entries (e.g. every h5_key, plus any SEPARATE_GROUP_BY_H5_KEY
    split as two separate bars) rather than a separate figure per
    dataset. Bar height is the mean of that column across that entry's
    sessions; each session's own value is overlaid as a jittered dot, colored
    by its label -- one color palette shared across every label value seen in
    ANY entry (not per-entry), so the same label means the same color in
    every bar/panel, even across different datasets' metadata columns.

    entries : list of (x_label, dims (n_sessions, n_cols), labels or None)
    """
    n_cols = len(dim_columns)
    all_labels = sorted({str(l) for _, _, labels in entries if labels is not None for l in labels})
    cmap = plt.get_cmap("tab20")
    colors = {label: cmap(i % 20) for i, label in enumerate(all_labels)}

    f, ax_arr = plt.subplots(n_cols, 1, figsize=(max(10, len(entries) * 0.9), 5 * n_cols), squeeze=False, sharex=True)
    ax_arr = ax_arr[:, 0]
    rng = np.random.default_rng(0)
    x_labels = [x_label for x_label, _, _ in entries]

    for col_idx, col_name in enumerate(dim_columns):
        ax = ax_arr[col_idx]
        bar_heights = [dims[:, col_idx].mean() if len(dims) else 0.0 for _, dims, _ in entries]
        ax.bar(range(len(entries)), bar_heights, color="lightgray", zorder=1)

        for x, (_, dims, labels) in enumerate(entries):
            if len(dims) == 0:
                continue
            values = dims[:, col_idx]
            jitter = rng.uniform(-0.3, 0.3, size=len(values))
            point_colors = "black" if labels is None else [colors[str(l)] for l in labels]
            ax.scatter(
                np.full(len(values), x) + jitter, values,
                color=point_colors, s=10, zorder=2, edgecolors="none",
            )

        ax.set_xticks(range(len(entries)))
        # Only the bottom panel shows x-tick labels (sharex means they'd
        # otherwise repeat, rotated, on every panel and clip into the one
        # above it since the stacked panels sit close together).
        if col_idx == n_cols - 1:
            ax.set_xticklabels(x_labels, rotation=60, ha="right")
        ax.set_ylabel(col_name)

    if all_labels:
        handles = [
            plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=colors[label], label=label, markersize=6)
            for label in all_labels
        ]
        f.legend(handles=handles, bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=7, ncol=max(1, len(all_labels) // 25))

    f.suptitle(title)
    f.tight_layout()
    f.savefig(out_path, bbox_inches="tight")
    plt.close(f)


def main():
    assert not (USE_MIDFWD_ROTATION and USE_SNOUT_TAILBASE_DIST), (
        "USE_SNOUT_TAILBASE_DIST is only meaningful when USE_MIDFWD_ROTATION is False "
        "(session_dims_for_windows always measures the 3-dim bounding-box range)."
    )

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    if USE_MIDFWD_ROTATION:
        mode_suffix, mode_title = "_midfwd", " (midfwd-rotated 2s windows)"
        dim_columns = ["dim0_range", "dim1_range", "dim2_range"]
    elif USE_SNOUT_TAILBASE_DIST:
        mode_suffix, mode_title = "_snouttailbase", " (snout-tailbase distance)"
        dim_columns = ["snout_tailbase_dist"]
    else:
        mode_suffix, mode_title = "", ""
        dim_columns = ["dim0_range", "dim1_range", "dim2_range"]

    if SCALE_BODY_LENGTH:
        mode_suffix += "_scaled"
        mode_title += " (scaled body length)"

    keypoint_pairs = {}
    if USE_SNOUT_TAILBASE_DIST:
        for name in DATASETS:
            assert name in SNOUT_TAILBASE_KEYPOINTS, (
                f"[{name}] missing from SNOUT_TAILBASE_KEYPOINTS -- add its "
                "(snout_idx, tailbase_idx) pair."
            )
            keypoint_pairs[name] = SNOUT_TAILBASE_KEYPOINTS[name]

    dataset = RodentDiskDataset(
        data_path=DATA_PATH,
        dataset_names=DATASETS,
        train_val_test="train",
        strides={name: 1 for name in DATASETS},
        min_interp_fps=25,
        balanced_sampling=False,
        virtual_rodent_x_splits=["base", "xsession", "xsubject"],
        use_speed_filtered_inds=False,
        scale_body=SCALE_BODY_LENGTH,
        # Neither is used anywhere in this script, and use_default_seg_len's
        # default (True) would crash -- OFFSETS_3D has no "mouse_ephys" or
        # "sdannce" entry (only "mouse44_ephys", "pairr24m", "virtual_rodent",
        # "wu_iclr25").
        use_pe_indices=False,
        use_default_seg_len=False,
    )

    entries = []  # (x_label, dims, labels) across every h5_key/subset, for the combined bar+dots plot
    session_dims_by_key = {}  # group_source_key -> dims, for compute_dataset_means

    for name in DATASETS:
        meta, ratmeta_key = None, None
        if name in RATMETA_BY_DATASET:
            meta_path, ratmeta_key, index_col = RATMETA_BY_DATASET[name]
            meta = pd.read_csv(Path(DATA_PATH) / meta_path, index_col=index_col)

        for h5_key in dataset.dataset_to_h5_keys[name]:
            print(f"Processing {h5_key}...")
            ids = np.asarray(dataset.h5_ids[h5_key])  # load once -- reused for every session's mask below
            pose_h5 = dataset.h5_datasets[h5_key]

            # Restrict to sessions with at least one train-split window --
            # dataset.h5_ids/h5_datasets are the RAW, unfiltered per-file
            # arrays (unaffected by the train/val/test split), so a session
            # with zero surviving windows (e.g. pairr24m_a2 has 14 sessions
            # where that agent's pose is entirely NaN -- only agent 1 was
            # tracked -- and every one of their windows was correctly dropped
            # by the split) still shows up in np.unique(ids). Computing a
            # percentile over such a session's raw (all-NaN) frames just
            # produces NaN, which then poisons compute_dataset_means's pooled
            # mean too -- skip sessions the actual training pipeline would
            # never use anyway.
            all_sessions = np.unique(ids)
            window_counts_all = session_window_counts(dataset, h5_key, all_sessions)
            has_windows = window_counts_all > 0
            unique_sessions = all_sessions[has_windows]
            window_counts = window_counts_all[has_windows]
            if not has_windows.all():
                print(
                    f"[{name}] {h5_key}: skipping {(~has_windows).sum()} session(s) "
                    f"with zero train-split windows (likely all-NaN for this h5_key)"
                )

            if USE_MIDFWD_ROTATION:
                session_dims = session_dims_for_windows(dataset, h5_key, name, unique_sessions)
            else:
                session_dims = session_dims_for(
                    pose_h5, ids, unique_sessions, keypoint_pair=keypoint_pairs.get(name)
                )

            labels = None
            if meta is not None:
                # .loc[unique_sessions] (not .isin) so rows come back in
                # exactly unique_sessions's order -- the same order
                # session_dims is built in above, so they line up row-for-row.
                labels = meta.loc[unique_sessions][ratmeta_key].to_numpy()

            base_out_name = f"session_dim_ranges_{h5_key}{mode_suffix}"

            def _save(dims, lbls, sess_ids, counts, x_label, name_suffix="", group_source_key=None):
                out_name = base_out_name
                if lbls is not None:
                    out_name += f"_{ratmeta_key}"
                out_name += name_suffix.replace(" ", "_").replace(".", "")

                table = pd.DataFrame(dims, columns=dim_columns)
                table.insert(0, "session_id", sess_ids)
                table["n_windows"] = counts
                if lbls is not None:
                    table[ratmeta_key] = lbls
                table.to_csv(FIGURES_DIR / f"{out_name}.csv", index=False)

                entries.append((x_label, dims, lbls))
                if group_source_key is not None:
                    session_dims_by_key[group_source_key] = dims

            separate_value = SEPARATE_GROUP_BY_H5_KEY.get(h5_key)
            if separate_value is not None and labels is not None:
                separate_mask = labels == separate_value
                _save(
                    session_dims[separate_mask], labels[separate_mask],
                    unique_sessions[separate_mask], window_counts[separate_mask],
                    f"{h5_key} ({separate_value})", f" ({separate_value}_only)",
                    group_source_key=f"{h5_key}|{separate_value}",
                )
                _save(
                    session_dims[~separate_mask], labels[~separate_mask],
                    unique_sessions[~separate_mask], window_counts[~separate_mask],
                    f"{h5_key} (excl. {separate_value})", f" (excl_{separate_value})",
                    group_source_key=f"{h5_key}|excl_{separate_value}",
                )
            else:
                _save(
                    session_dims, labels, unique_sessions, window_counts, h5_key,
                    group_source_key=h5_key,
                )

    dataset.close()

    plot_bar_with_dots(
        entries, dim_columns,
        f"Session Dim Ranges by h5_key{mode_title}",
        FIGURES_DIR / f"session_dims_by_h5_key{mode_suffix}.png",
    )

    means_table = compute_dataset_means(session_dims_by_key, dim_columns)
    means_table.to_csv(FIGURES_DIR / f"dataset_means{mode_suffix}.csv", index=False)
    print(means_table.to_string(index=False))


if __name__ == "__main__":
    main()
