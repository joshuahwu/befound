"""
One-off migration: find every existing "*_2s.npy" train/val/test window-index
file for sdannce, pairr24m, wu_iclr25, and virtual_rodent (written by an
earlier version of generate_train_val_test_splits.py that stored full
(n_windows, window_size) frame-index arrays) and overwrite each one in place
with just its center column (n_windows,) -- a window is window_size
consecutive integers, so the center index plus the per-dataset window size
(befound.data.constants.get_window_size) is enough for RodentDiskDataset to
reconstruct the full window at load time. This is a window_size-factor
reduction in file size, matching the format generate_train_val_test_splits.py
now writes for new splits.

Files already in the new (n_windows,) center-only format are left untouched
(idempotent, safe to re-run). Only "_2s"-tagged files are touched -- the
original, un-suffixed split files are never read or written by this script.
"""

import argparse
from pathlib import Path

import numpy as np

DATA_PATH = "/mnt/home/jwu10/working/ceph/data/"
DATASETS = ["sdannce", "pairr24m", "wu_iclr25", "virtual_rodent"]

def _find_inds_files(dataset_dir: Path):
    """Every "*_2s.npy" file under dataset_dir whose name contains "inds" --
    covers plain {split}_inds_2s.npy, suffixed variants (e.g.
    {split}_inds_social_2s.npy, {split}_inds_xsubject_2s.npy), and wu_iclr25's
    parkinsons/{split}/inds_2s.npy."""
    return sorted(p for p in dataset_dir.rglob("*_2s.npy") if "inds" in p.name)


def convert_file(path: Path, dry_run: bool = False) -> str:
    arr = np.load(path)
    before_bytes = path.stat().st_size

    if arr.ndim == 1:
        return f"  {path}: already center-only ({arr.shape}), skipping"

    if arr.ndim != 2:
        return f"  {path}: unexpected ndim={arr.ndim}, shape={arr.shape}, skipping"

    before_shape = arr.shape
    center = arr[:, arr.shape[1] // 2].copy()
    del arr
    if not dry_run:
        np.save(path, center)
    after_bytes = center.nbytes + 128  # npy header overhead, approximate

    return (
        f"  {path}: {before_shape} -> {center.shape} "
        f"({before_bytes / 1e6:.1f}MB -> ~{after_bytes / 1e6:.1f}MB)"
        + (" [dry-run, not written]" if dry_run else "")
    )

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-path", default=DATA_PATH)
    ap.add_argument("--datasets", nargs="+", default=DATASETS, choices=DATASETS)
    ap.add_argument(
        "--dry-run", action="store_true", help="Report what would change without writing anything"
    )
    args = ap.parse_args()

    for name in args.datasets:
        dataset_dir = Path(args.data_path) / name
        if not dataset_dir.exists():
            print(f"[{name}] {dataset_dir} does not exist, skipping")
            continue

        files = _find_inds_files(dataset_dir)
        if not files:
            print(f"[{name}] no *_2s.npy inds files found under {dataset_dir}")
            continue

        print(f"[{name}] {len(files)} file(s) found under {dataset_dir}")
        for path in files:
            print(convert_file(path, dry_run=args.dry_run))


if __name__ == "__main__":
    main()
