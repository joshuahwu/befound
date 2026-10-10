"""
Generate train/val/test window-index splits for each dataset's pose h5 file(s).

Wraps befound.data.datasets.train_val_test_split, which now takes an existing
pose h5 path (RodentDiskDataset's on-disk format: "pose" + per-frame "ids")
rather than an in-memory pose_dict, and no longer writes any metadata -- just
the split's own train/val/test index arrays.

Each saved *_inds_2s.npy file holds only the *center* frame index of every
window (shape (n_windows,), not (n_windows, window_size)) -- a window is just
window_size consecutive integers, so storing all of them on disk is storing
window_size near-duplicate copies of the same information per row.
RodentDiskDataset reconstructs the full window at load time from the center
index plus the per-dataset window size (befound.data.constants.get_window_size,
a pure function of DATA_FPS -- the same one this script uses to build windows
in the first place, so window size is always derived, never passed between
the two). This is a window_size-factor reduction in file size and load time
over storing full windows (e.g. sdannce's window is ~100-200 frames).

Per-dataset conventions (matched to what RodentDiskDataset / the preprocess_*
loaders in befound.data.datasets already expect on disk, extended with a
"_2s" tag since this script never overwrites the original, already-committed
split files -- it only ever adds new ones alongside them):
  - mouse_ephys                 : single pose.h5      -> {split}_inds_2s.npy
  - virtual_rodent               : pose.h5 split into pose_base.h5/pose_xsession.h5/
                                   pose_xsubject.h5 (bespoke per-animal grouping, see below),
                                   each independently -> {split}_inds_2s.npy /
                                   {split}_inds_xsession_2s.npy / {split}_inds_xsubject_2s.npy
  - pairr24m                    : pose_a1.h5/pose_a2.h5 (paired, shared split)
                                   -> {split}_inds_social_2s.npy
                                   pose_single.h5 (sessions where only one
                                   agent was ever tracked -- see
                                   ceph/data/pairr24m/code/split_pose_single.py;
                                   separate split)
                                   -> {split}_inds_single_2s.npy
  - sdannce                     : pose_a1.h5/pose_a2.h5 (paired, shared split)
                                   -> {split}_inds_social_2s.npy
                                   pose_single.h5 (separate split)
                                   -> {split}_inds_single_2s.npy
  - wu_iclr25                   : parkinsons/pose.h5    -> parkinsons/{split}/inds_2s.npy

Paired datasets (pairr24m's and sdannce's a1/a2) are split once from one
agent's ids and reused for both -- this requires both agents' h5 files to
share the exact same per-frame ids, which is asserted before saving.

wu_iclr25 is handled differently from the rest: rather than re-running the
random block split (which would reassign which raw frames are train/val/test),
it reads the frame membership already fixed by the existing
parkinsons/{split}/inds.npy files and just re-windows those same frames at the
new, longer window size -- any new window that would straddle two splits (or
touch a frame that wasn't in any split originally, e.g. dropped by the
original `gap` margin) is dropped rather than assigned.

virtual_rodent also gets bespoke, self-contained per-animal handling (reading
session->animal from meta.csv and session boundaries from pose.h5's ids --
neither file is ever modified). Every session is assigned to exactly one of
three groups (virtual_rodent_session_groups, seeded by --seed):
  - "xsubject": coltrane and gerry, entirely -- held-out subjects
  - "xsession": 25% (floor) of art/bud/duke/freddie's own sessions -- held-out sessions
  - "regular": everything else (remaining 75% of art/bud/duke/freddie, all of espie)
pose.h5 is first split into pose_base.h5/pose_xsession.h5/pose_xsubject.h5
(one per group, via neuroposelib.io.write.pose_h5), then the standard
train_val_test_split runs independently on each -- same as every other
dataset, just three of them instead of one. Every group gets its own
50/25/25 train/val/test block split, except espie, which is kept 100% train
via train_val_test_split's force_train_ids.

By default, a dataset (or single/social/wu_iclr25 sub-target) is skipped if
its "_2s" split files already exist. Pass --overwrite to regenerate those (the
original, un-suffixed split files are never written to by this script).
--datasets restricts which targets to run.
"""

import argparse
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view
from neuroposelib import read, write

from befound.data.constants import DATA_FPS, get_window_size
from befound.data.datasets import train_val_test_split, _has_any_nan, _drop_nan_windows

DATA_PATH = "/mnt/home/jwu10/working/ceph/data/"
MODEL_CONFIG_PATH = (
    "/mnt/home/jwu10/working/ceph/results/foundation/multi_dataset/base/model_config.yaml"
)

SPLITS = ["train", "val", "test"]
TAG = "_2s"  # never overwrite the original split files; write alongside them

# Fixed regardless of dataset -- not exposed as CLI options.
TRAIN_FRAC, VAL_FRAC, TEST_FRAC = 0.5, 0.25, 0.25
BLOCK_SECONDS = 60  # block_size = ~1 minute of frames, using each dataset's fps
GAP_SECONDS = 0.5  # gap = ~0.5s of frames, using each dataset's fps
STRIDE = 1

DEFAULT_SPLIT_KWARGS = dict(
    seed=0,
)


def _block_size(target_name):
    """~1 minute of frames at this dataset's fps (befound.data.constants.DATA_FPS)."""
    return round(BLOCK_SECONDS * DATA_FPS[target_name])


def _gap(target_name):
    """~0.5s of frames at this dataset's fps (befound.data.constants.DATA_FPS)."""
    return round(GAP_SECONDS * DATA_FPS[target_name])


def _load_ids(h5_path):
    with h5py.File(h5_path, "r") as hf:
        return np.asarray(hf["ids"])


def _session_ranges(pose_h5_path):
    """(session_offsets, session_lengths, session_ids, n_frames), mirroring the
    contiguous-per-session assumption train_val_test_split makes from the h5's
    ids. session_ids are the actual per-frame id values (e.g. meta.csv's
    session_id), in temporal order."""
    ids = _load_ids(pose_h5_path)
    unique_ids, first_idx, lengths = np.unique(ids, return_index=True, return_counts=True)
    order = np.argsort(first_idx)
    return first_idx[order], lengths[order], unique_ids[order], len(ids)


def _split_files_exist(save_dir, suffix):
    save_dir = Path(save_dir)
    return all((save_dir / f"{split}_inds{suffix}{TAG}.npy").exists() for split in SPLITS)


def _center_column(inds):
    """A window is just window_size consecutive integers, so storing all of
    them is storing window_size near-duplicate copies of the same
    information per row -- save only the center index; RodentDiskDataset
    reconstructs the full window at load time from that plus the per-dataset
    window size (befound.data.constants.get_window_size), which is far
    cheaper to write/read than the full 2D array (window_size x smaller)."""
    return inds[:, inds.shape[1] // 2]


def _save_splits(save_dir, suffix, train_inds, val_inds, test_inds):
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    for split, inds in zip(SPLITS, [train_inds, val_inds, test_inds]):
        np.save(save_dir / f"{split}_inds{suffix}{TAG}.npy", _center_column(inds))


def generate_simple_split(name, pose_h5, save_dir, suffix="", overwrite=False, **split_kwargs):
    """Single pose.h5 -> {split}_inds{suffix}_2s.npy in save_dir."""
    if not overwrite and _split_files_exist(save_dir, suffix):
        print(f"[{name}] {TAG} split files already exist in {save_dir} (suffix='{suffix}'), skipping.")
        return

    print(f"[{name}] generating split from {pose_h5} ...")
    train_inds, val_inds, test_inds = train_val_test_split(pose_h5, **split_kwargs)

    assert train_inds.shape[0] == len(np.unique(train_inds[:, 0]))
    assert val_inds.shape[0] == len(np.unique(val_inds[:, 0]))
    assert test_inds.shape[0] == len(np.unique(test_inds[:, 0]))

    assert np.all(np.diff(train_inds, axis=1) == 1)
    assert np.all(np.diff(test_inds, axis=1) == 1)
    assert np.all(np.diff(val_inds, axis=1) == 1)

    _save_splits(save_dir, suffix, train_inds, val_inds, test_inds)
    print(
        f"[{name}] saved {len(train_inds)} train / {len(val_inds)} val / "
        f"{len(test_inds)} test windows to {save_dir} (suffix='{suffix}{TAG}')"
    )

def generate_paired_split(name, pose_h5_a1, pose_h5_a2, save_dir, suffix="", overwrite=False, **split_kwargs):
    """Split from agent 1's ids, verify agent 2 shares identical ids, save once
    for both to reuse (matches how RodentDiskDataset loads one inds file for
    both pose_a1 and pose_a2)."""
    if not overwrite and _split_files_exist(save_dir, suffix):
        print(f"[{name}] {TAG} split files already exist in {save_dir} (suffix='{suffix}'), skipping.")
        return

    ids_a1 = _load_ids(pose_h5_a1)
    ids_a2 = _load_ids(pose_h5_a2)
    assert np.array_equal(ids_a1, ids_a2), (
        f"[{name}] pose_a1 and pose_a2 ids differ -- can't safely share a single split. "
        f"({pose_h5_a1} vs {pose_h5_a2})"
    )

    print(f"[{name}] generating shared split from {pose_h5_a1} (verified pose_a2 matches) ...")
    train_inds, val_inds, test_inds = train_val_test_split(pose_h5_a1, **split_kwargs)

    assert train_inds.shape[0] == len(np.unique(train_inds[:, 0]))
    assert val_inds.shape[0] == len(np.unique(val_inds[:, 0]))
    assert test_inds.shape[0] == len(np.unique(test_inds[:, 0]))

    assert np.all(np.diff(train_inds, axis=1) == 1)
    assert np.all(np.diff(test_inds, axis=1) == 1)
    assert np.all(np.diff(val_inds, axis=1) == 1)

    # train_val_test_split already dropped windows with a NaN in pose_a1; also
    # drop any window with a NaN in pose_a2, since both agents share this
    # split -- a window is excluded if *either* agent has a NaN in it.
    with h5py.File(pose_h5_a2, "r") as hf:
        pose_dataset_a2 = hf["pose"]
        if _has_any_nan(pose_dataset_a2):
            n_train, n_val, n_test = len(train_inds), len(val_inds), len(test_inds)
            train_inds = _drop_nan_windows(pose_dataset_a2, train_inds)
            val_inds = _drop_nan_windows(pose_dataset_a2, val_inds)
            test_inds = _drop_nan_windows(pose_dataset_a2, test_inds)
            print(
                f"[{name}] dropped additional NaN windows from pose_a2: "
                f"{n_train - len(train_inds)} train, {n_val - len(val_inds)} val, "
                f"{n_test - len(test_inds)} test"
            )
        else:
            print(f"[{name}] no NaNs found in pose_a2; no additional windows dropped.")

    _save_splits(save_dir, suffix, train_inds, val_inds, test_inds)
    print(
        f"[{name}] saved {len(train_inds)} train / {len(val_inds)} val / "
        f"{len(test_inds)} test windows to {save_dir} (suffix='{suffix}{TAG}')"
    )


XSUBJECT_ANIMALS = {"coltrane", "gerry"}
REGULAR_ONLY_ANIMALS = {"espie"}

# Group -> {split}_inds{suffix}_2s.npy suffix, and group -> its own split pose
# file. Shared by virtual_rodent_session_groups/generate_virtual_rodent_split.
VIRTUAL_RODENT_GROUP_SUFFIX = {"regular": "", "xsession": "_xsession", "xsubject": "_xsubject"}
VIRTUAL_RODENT_GROUP_TO_FILENAME = {
    "regular": "pose_base.h5",
    "xsession": "pose_xsession.h5",
    "xsubject": "pose_xsubject.h5",
}


def virtual_rodent_session_groups(meta, seed):
    """Assign every virtual_rodent session to exactly one of three groups:
      - "xsubject": coltrane and gerry, entirely -- held-out subjects
      - "xsession": 25% (floor) of art/bud/duke/freddie's own sessions -- held-out sessions
      - "regular": everything else (remaining 75% of art/bud/duke/freddie, all of espie)
    Deterministic given `meta` and `seed` -- used by generate_virtual_rodent_split
    both to split pose.h5 into pose_base.h5/pose_xsession.h5/pose_xsubject.h5
    and to name each group's saved {split}_inds*_2s.npy files, so the two
    always agree on which sessions belong to which group.

    Parameters
    ----------
    meta : pd.DataFrame
        virtual_rodent/meta.csv, indexed by session id, with an "animal" column.
    seed : int
        RNG seed (should match whatever generate_virtual_rodent_split was run
        with, e.g. --seed / DEFAULT_SPLIT_KWARGS["seed"]).

    Returns
    -------
    dict mapping session id -> "regular" | "xsession" | "xsubject"
    """
    rng = np.random.default_rng(seed)
    sessions_by_animal = {a: g.index.tolist() for a, g in meta.groupby("animal")}

    session_group = {}
    for animal, sess_ids in sessions_by_animal.items():
        if animal in XSUBJECT_ANIMALS:
            for s in sess_ids:
                session_group[s] = "xsubject"
        elif animal in REGULAR_ONLY_ANIMALS:
            for s in sess_ids:
                session_group[s] = "regular"
        else:  # art, bud, duke, freddie
            shuffled = rng.permutation(sess_ids)
            n_xsession = len(shuffled) // 4  # 25%, rounded down
            for s in shuffled[:n_xsession]:
                session_group[s] = "xsession"
            for s in shuffled[n_xsession:]:
                session_group[s] = "regular"

    return session_group


def generate_virtual_rodent_split(data_path, overwrite=False, **split_kwargs):
    """virtual_rodent's bespoke per-animal split. Every session is assigned to
    exactly one of three groups (virtual_rodent_session_groups, seeded by
    split_kwargs["seed"]):
      - "xsubject": coltrane and gerry, entirely -- held-out subjects
      - "xsession": 25% (floor) of art/bud/duke/freddie's own sessions -- held-out sessions
      - "regular": everything else (remaining 75% of art/bud/duke/freddie, all of espie)

    Unlike the other datasets, this first splits pose.h5 itself into three
    separate pose files -- pose_base.h5 (regular), pose_xsession.h5,
    pose_xsubject.h5 (via neuroposelib.io.write.pose_h5) -- so each group's
    frames are contiguously renumbered from 0 in their own file. The standard
    train_val_test_split then runs independently on each of those three
    files, exactly like every other dataset: no bespoke block-splitting or
    index remapping needed, since each pose file only ever holds its own
    group's frames.

    Espie is the one animal always in "regular" and is kept 100% train (never
    val/test): train_val_test_split's force_train_ids does this during the
    block split itself (every kept window in an espie session goes straight
    to train, skipping the random per-session block assignment).

    pose.h5 and meta.csv are only ever read, never modified. Writes
    pose_base.h5/pose_xsession.h5/pose_xsubject.h5 and
    {split}_inds_2s.npy / {split}_inds_xsession_2s.npy / {split}_inds_xsubject_2s.npy
    in virtual_rodent/.
    """
    name = "virtual_rodent"
    base_dir = f"{data_path}{name}/"
    pose_h5_path = f"{base_dir}pose.h5"
    out_paths = {g: Path(base_dir + fn) for g, fn in VIRTUAL_RODENT_GROUP_TO_FILENAME.items()}

    if not overwrite and all(_split_files_exist(base_dir, s) for s in VIRTUAL_RODENT_GROUP_SUFFIX.values()):
        print(f"[{name}] {TAG} split files already exist in {base_dir}, skipping.")
        return

    meta = pd.read_csv(f"{base_dir}meta.csv", index_col=0)
    session_group = virtual_rodent_session_groups(meta, split_kwargs["seed"])
    espie_sessions = set(meta.index[meta["animal"] == "espie"])

    session_ids_by_group = {g: [] for g in VIRTUAL_RODENT_GROUP_TO_FILENAME}
    for sid, group in session_group.items():
        session_ids_by_group[group].append(sid)

    # Split pose.h5 into one file per group: each unique value in "ids" is an
    # entire session, so a boolean mask built from that group's session ids
    # directly selects exactly its frames, in their original relative order
    # (each session's own frames stay contiguous, since they already are in
    # pose.h5 -- train_val_test_split, run per group below, depends on that).
    # Boolean-indexing pose_in (an h5py Dataset) reads only the selected
    # frames, same as the old per-session slicing, just without needing to
    # track each session's own offset/length first.
    mask_check = []
    with h5py.File(pose_h5_path, "r") as hf:
        pose_in = hf["pose"]
        ids_in = np.asarray(hf["ids"])
        n_frames = len(ids_in)

        for group, out_path in out_paths.items():
            group_session_ids = session_ids_by_group[group]
            if not group_session_ids:
                print(f"[{name}] [{group}] 0 sessions selected, skipping {out_path.name}")
                continue

            mask = np.isin(ids_in, group_session_ids)
            mask_check.append(mask[:, None])
            group_pose = pose_in[mask]
            group_ids = ids_in[mask]
            write.pose_h5(group_pose, group_ids, str(out_path))
            print(
                f"[{name}] [{group}] {len(group_session_ids)} session(s), {len(group_ids)} of {n_frames} frames -> {out_path}"
            )

    # Per-frame count of how many groups' masks claim it -- should be exactly
    # 1 everywhere (no overlaps, and every frame lands in some group).
    frame_group_counts = np.concatenate(mask_check, axis=-1).sum(axis=-1)
    assert (frame_group_counts == 1).all(), (
        f"{(frame_group_counts != 1).sum()} frame(s) claimed by != 1 group "
        "-- masks overlap or some frame isn't covered by any group"
    )

    # Run the standard split independently on each group's own pose file.
    for group, suffix in VIRTUAL_RODENT_GROUP_SUFFIX.items():
        out_path = out_paths[group]
        force_train_ids = espie_sessions if group == "regular" else None
        train_inds, val_inds, test_inds = train_val_test_split(
            str(out_path), force_train_ids=force_train_ids, **split_kwargs
        )
        len(np.unique(train_inds))

        _save_splits(base_dir, suffix, train_inds, val_inds, test_inds)
        print(
            f"[{name}] [{group}] saved {len(train_inds)} train / {len(val_inds)} val / "
            f"{len(test_inds)} test windows to {base_dir} (suffix='{suffix}{TAG}')"
        )


def generate_wu_iclr25_split(data_path, overwrite=False, **split_kwargs):
    """wu_iclr25 stores indices as parkinsons/{split}/inds.npy, one file per
    split subfolder. Rather than re-running the random block split (which
    would reassign which raw frames land in train/val/test), this reads the
    frame membership already fixed by those existing inds.npy files and only
    re-windows those same frames at the new (longer) window size and stride=1,
    respecting session boundaries. Any new window that would straddle two
    splits, or touch a frame not originally assigned to any split (e.g.
    trimmed by the original `gap` margin), is dropped rather than assigned.
    Writes parkinsons/{split}/inds_2s.npy; never touches inds.npy.
    """
    pose_h5 = f"{data_path}wu_iclr25/parkinsons/pose.h5"
    split_dirs = {
        split: Path(f"{data_path}wu_iclr25/parkinsons/{split}") for split in SPLITS
    }
    out_paths = {split: split_dirs[split] / f"inds{TAG}.npy" for split in SPLITS}

    if not overwrite and all(p.exists() for p in out_paths.values()):
        print(f"[wu_iclr25] inds{TAG}.npy already exist in all split folders, skipping.")
        return

    for split, d in split_dirs.items():
        assert (d / "inds.npy").exists(), (
            f"[wu_iclr25] missing existing {d}/inds.npy to derive frame membership from"
        )

    window = split_kwargs["window"]
    stride = split_kwargs["stride"]

    offsets, lengths, _session_ids, n_frames = _session_ranges(pose_h5)

    # Frame -> split index (0=train, 1=val, 2=test), -1 = not in any split
    frame_split = np.full(n_frames, -1, dtype=np.int8)
    for split_idx, split in enumerate(SPLITS):
        old_inds = np.load(split_dirs[split] / "inds.npy")
        frames = np.unique(old_inds.reshape(-1))
        assert (frame_split[frames] == -1).all(), (
            f"[wu_iclr25] some frame(s) already assigned to another split "
            f"while reading {split}/inds.npy -- existing splits overlap?"
        )
        frame_split[frames] = split_idx

    new_inds = {split: [] for split in SPLITS}
    for start, n in zip(offsets, lengths):
        if n < window:
            continue
        frame_idx = np.arange(start, start + n)
        session_windows = sliding_window_view(frame_idx, window)[::stride]

        window_splits = frame_split[session_windows]  # (n_windows, window)
        first_col = window_splits[:, 0]
        # keep only windows that are entirely inside one (already-assigned) split
        uniform = (window_splits == first_col[:, None]).all(axis=1)
        valid = uniform & (first_col >= 0)

        for split_idx, split in enumerate(SPLITS):
            new_inds[split].append(session_windows[valid & (first_col == split_idx)])

    new_inds = {split: np.concatenate(new_inds[split], axis=0) for split in SPLITS}
    new_inds = {split: new_inds[split][np.argsort(new_inds[split][:, 0])] for split in SPLITS}
    assert np.intersect1d(new_inds["train"].flatten(), new_inds["val"].flatten()).size == 0
    assert np.intersect1d(new_inds["train"].flatten(), new_inds["test"].flatten()).size == 0
    assert np.intersect1d(new_inds["val"].flatten(), new_inds["test"].flatten()).size == 0

    assert new_inds["train"].shape[0] == len(np.unique(new_inds["train"][:, 0]))
    assert new_inds["val"].shape[0] == len(np.unique(new_inds["val"][:, 0]))
    assert new_inds["test"].shape[0] == len(np.unique(new_inds["test"][:, 0]))

    assert np.all(np.diff(new_inds["train"], axis=1) == 1)
    assert np.all(np.diff(new_inds["test"], axis=1) == 1)
    assert np.all(np.diff(new_inds["val"], axis=1) == 1)
    
    for split in SPLITS:
        inds = new_inds[split]
        # inds = np.concatenate(new_inds[split], axis=0)
        #     if new_inds[split]
        #     else np.empty((0, window), dtype=np.int64)
        # )
        # inds = inds[np.argsort(inds[:, 0])]
        assert (np.diff(inds, axis=1) == 1).all(), (
            f"[wu_iclr25] some {split} windows are not contiguous after re-windowing -- something went wrong"
        )
        split_dirs[split].mkdir(parents=True, exist_ok=True)
        np.save(out_paths[split], _center_column(inds))
        print(f"[wu_iclr25] saved {len(inds)} {split} windows to {out_paths[split]}")


TARGETS = ["mouse_ephys", "virtual_rodent", "pairr24m", "sdannce", "wu_iclr25"]


def _kwargs_for(target_name, split_kwargs, model_window, min_interp_fps):
    """Per-dataset split kwargs: fixed train/val/test fracs and stride=1, plus
    block_size, window, and gap derived from that dataset's fps (window also
    depends on model_window/min_interp_fps, both read from the model config);
    everything else (seed) comes from CLI."""
    return dict(
        split_kwargs,
        window=get_window_size(target_name, model_window, min_interp_fps),
        stride=STRIDE,
        block_size=_block_size(target_name),
        gap=_gap(target_name),
        train_frac=TRAIN_FRAC,
        val_frac=VAL_FRAC,
        test_frac=TEST_FRAC,
    )


def run(data_path, targets, overwrite, split_kwargs, model_window, min_interp_fps):
    kw = lambda name: _kwargs_for(name, split_kwargs, model_window, min_interp_fps)

    if "mouse_ephys" in targets:
        generate_simple_split(
            "mouse_ephys",
            f"{data_path}mouse_ephys/pose.h5",
            f"{data_path}mouse_ephys/",
            overwrite=overwrite,
            **kw("mouse_ephys"),
        )

    if "virtual_rodent" in targets:
        generate_virtual_rodent_split(
            data_path, overwrite=overwrite, **kw("virtual_rodent")
        )

    if "pairr24m" in targets:
        generate_paired_split(
            "pairr24m_social",
            f"{data_path}pairr24m/pose_a1.h5",
            f"{data_path}pairr24m/pose_a2.h5",
            f"{data_path}pairr24m/",
            suffix="_social",
            overwrite=overwrite,
            **kw("pairr24m"),
        )
        generate_simple_split(
            "pairr24m_single",
            f"{data_path}pairr24m/pose_single.h5",
            f"{data_path}pairr24m/",
            suffix="_single",
            overwrite=overwrite,
            **kw("pairr24m"),
        )

    if "sdannce" in targets:
        generate_paired_split(
            "sdannce_social",
            f"{data_path}sdannce/pose_a1.h5",
            f"{data_path}sdannce/pose_a2.h5",
            f"{data_path}sdannce/",
            suffix="_social",
            overwrite=overwrite,
            **kw("sdannce"),
        )
        generate_simple_split(
            "sdannce_single",
            f"{data_path}sdannce/pose_single.h5",
            f"{data_path}sdannce/",
            suffix="_single",
            overwrite=overwrite,
            **kw("sdannce"),
        )

    if "wu_iclr25" in targets:
        generate_wu_iclr25_split(
            data_path, overwrite=overwrite, **kw("wu_iclr25")
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-path", default=DATA_PATH)
    parser.add_argument(
        "--model-config",
        default=MODEL_CONFIG_PATH,
        help="model_config.yaml providing model.window and train.augmentations.fps.min_fps",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=TARGETS,
        choices=TARGETS,
        help="Which dataset(s) to (re)generate splits for.",
    )
    overwrite=True
    seed=0
    args = parser.parse_args()

    config = read.config(args.model_config)
    model_window = config["model"]["window"]
    min_interp_fps = config["train"]["augmentations"]["fps"]["min_fps"]

    split_kwargs = dict(
        seed=seed,
    )

    run(
        args.data_path,
        args.datasets,
        overwrite,
        split_kwargs,
        model_window,
        min_interp_fps,
    )
