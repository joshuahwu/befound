"""
Check MABe22 2-D keypoint data for anomalies **after** prepare_batch_2d_bespoke.

After prepare_batch_2d_bespoke, x2d is:
  - Root-centred: centre-frame kpt-0 is always (0,0)  <-- NOT an anomaly
  - Forward-aligned: centre-frame kpt-1 points along +x
  - Normalised by offsets_sum

Anomaly checks on x2d:
  1. (0, 0) keypoints on frames/keypoints OTHER than the expected centre root
  2. Large inter-frame jumps (in normalised units)

Usage:
  python check_data_2d.py --data_path /path/to/mabe22 [--split train] \
      [--jump_thresh 0.3] [--window 51] [--stride 6] [--batch_size 256]
"""

import argparse

import numpy as np
import torch
from torch.utils.data import DataLoader

from befound.get.get import get_mabe22_data
from befound.data import MabeWindowDataset
from befound.data.train_utils import prepare_batch_2d_bespoke

# ── CLI ──────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="Check MABe22 x2d data quality")
parser.add_argument("--data_path", "-d", type=str,
                    default="/hpc/group/tdunn/yw789/befound_code/bams/data/mabe_90hz/",
                    help="Path to MABe22 data directory")
parser.add_argument("--split", type=str, default="train",
                    choices=["train", "submission", "all"])
parser.add_argument("--window",     type=int,   default=51)
parser.add_argument("--stride",     type=int,   default=51)
parser.add_argument("--batch_size", type=int,   default=256)
parser.add_argument("--num_workers",type=int,   default=2)
parser.add_argument("--jump_thresh", type=float, default=0.1,
                    help="Jump threshold in normalised units (after /offsets_sum)")
args = parser.parse_args()

# No-augmentation dict for prepare_batch_2d_bespoke
AUGMENT_NONE = {
    "single_ablation": None,
    "2d_td":           None,
    "kpt_shuffle":     None,
}

# ── Load raw keypoints & build dataset / dataloader ──────────────────────────
print(f"[1/3] Loading keypoints  (split='{args.split}') ...")
result = get_mabe22_data(args.data_path, split=args.split, reindex=True)
kp = result[0]          # (N_traj, T, K, 2)
print(f"      keypoints shape: {kp.shape}   dtype: {kp.dtype}")

print(f"[2/3] Building MabeWindowDataset  (window={args.window}, stride={args.stride}) ...")
dataset = MabeWindowDataset(kp, window=args.window, stride=args.stride, pad_mode="edge")
offsets_sum = dataset.offsets_sum
loader = DataLoader(
    dataset,
    batch_size=args.batch_size,
    shuffle=False,
    num_workers=args.num_workers,
    pin_memory=False,
    drop_last=False,
)
print(f"      dataset length : {len(dataset)}  |  n_batches: {len(loader)}")
print(f"      offsets_sum    : {offsets_sum}")

half_w = args.window // 2   # index of the centre frame in x2d

# ── Iterate and collect anomaly statistics ────────────────────────────────────
print(f"[3/3] Iterating over x2d (jump_thresh={args.jump_thresh} normalised units) ...")
print( "      NOTE: centre-frame kpt-0 == (0,0) is expected and excluded from zero check.")

total_windows     = 0
windows_with_zero = 0   # anomalous (0,0) outside expected centre-root
windows_with_jump = 0

zero_kpt_counts = np.zeros(dataset.n_keypts, dtype=np.int64)
jump_kpt_counts = np.zeros(dataset.n_keypts, dtype=np.int64)

zero_example = None   # (global_win_idx, x2d tensor)
jump_example = None   # (global_win_idx, x2d tensor, max_dist)

for batch_idx, batch in enumerate(loader):
    # ── prepare_batch_2d_bespoke: pose -> x2d (normalised, centred, fwd-aligned)
    data = prepare_batch_2d_bespoke(
        batch,
        augment_dict=AUGMENT_NONE,
        offsets_sum=offsets_sum,
        device="cpu",
    )
    x2d = data["x2d"]          # (B, W, K, 2)  on CPU, normalised
    B, W, K, _ = x2d.shape
    total_windows += B

    # ── (0, 0) check ──────────────────────────────────────────────────────
    # Build a mask that marks ALL (0,0) positions
    is_zero = (x2d[..., 0] == 0) & (x2d[..., 1] == 0)   # (B, W, K)

    # The centre frame's kpt-0 is *always* (0,0) by construction — exclude it
    is_zero[:, half_w, 0] = False

    has_zero = is_zero.any(dim=(1, 2))                    # (B,)
    windows_with_zero += int(has_zero.sum())
    zero_kpt_counts   += is_zero.any(dim=1).sum(dim=0).numpy()  # (K,)

    if zero_example is None and has_zero.any():
        idx = has_zero.nonzero(as_tuple=True)[0][0].item()
        zero_example = (batch_idx * args.batch_size + idx, x2d[idx].clone())

    # ── large jump check ──────────────────────────────────────────────────
    diff = x2d[:, 1:] - x2d[:, :-1]        # (B, W-1, K, 2)
    dist = diff.norm(dim=-1)                # (B, W-1, K)

    has_jump = (dist > args.jump_thresh).any(dim=(1, 2))  # (B,)
    windows_with_jump += int(has_jump.sum())
    jump_kpt_counts   += (dist > args.jump_thresh).any(dim=1).sum(dim=0).numpy()

    if jump_example is None and has_jump.any():
        idx = has_jump.nonzero(as_tuple=True)[0][0].item()
        jump_example = (batch_idx * args.batch_size + idx,
                        x2d[idx].clone(),
                        float(dist[idx].max()))

    if (batch_idx + 1) % 50 == 0:
        print(f"  batch {batch_idx+1}/{len(loader)} | "
              f"zero windows: {windows_with_zero} | "
              f"jump windows: {windows_with_jump}")

# ── Summary ──────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("ANOMALY SUMMARY  (x2d — normalised, centred, fwd-aligned)")
print("=" * 60)
print(f"offsets_sum             : {offsets_sum:.4f}")
print(f"Total windows checked   : {total_windows}")
print(f"Windows with (0,0) kpt  : {windows_with_zero}  "
      f"({100*windows_with_zero/max(total_windows, 1):.2f} %)  "
      f"[centre-frame kpt-0 excluded]")
print(f"Windows with jump >{args.jump_thresh:.2f} : {windows_with_jump}  "
      f"({100*windows_with_jump/max(total_windows, 1):.2f} %)")

print("\nPer-keypoint  (0,0) window counts  [centre kpt-0 excluded]:")
for k, cnt in enumerate(zero_kpt_counts):
    marker = "  <-- centre root (excluded)" if k == 0 else ""
    print(f"  kpt {k:2d}: {cnt}{marker}")

print(f"\nPer-keypoint  jump (>{args.jump_thresh}) window counts:")
for k, cnt in enumerate(jump_kpt_counts):
    print(f"  kpt {k:2d}: {cnt}")

if zero_example is not None:
    win_idx, win_tensor = zero_example
    print(f"\nFirst anomalous (0,0) example — global window index {win_idx}")
    print(f"  x2d shape: {tuple(win_tensor.shape)}")
    bad = (win_tensor[..., 0] == 0) & (win_tensor[..., 1] == 0)
    bad[half_w, 0] = False
    fr, kp_idx = bad.nonzero(as_tuple=True)
    for f, k in zip(fr[:8].tolist(), kp_idx[:8].tolist()):
        print(f"  frame {f:3d}  kpt {k:2d}  x2d={win_tensor[f, k].tolist()}"
              f"  {'<-- centre frame' if f == half_w else ''}")
else:
    print("\nNo anomalous (0,0) keypoints found.")

if jump_example is not None:
    win_idx, win_tensor, max_dist = jump_example
    print(f"\nFirst large-jump example — global window index {win_idx}  "
          f"max_dist={max_dist:.4f}")
    diff = win_tensor[1:] - win_tensor[:-1]    # (W-1, K, 2)
    dist = diff.norm(dim=-1)                    # (W-1, K)
    fr, kp_idx = (dist > args.jump_thresh).nonzero(as_tuple=True)
    for f, k in zip(fr[:8].tolist(), kp_idx[:8].tolist()):
        print(f"  frame {f:3d}->{f+1}  kpt {k:2d}  "
              f"dist={dist[f, k]:.4f}  "
              f"from {[round(v,4) for v in win_tensor[f, k].tolist()]} "
              f"to {[round(v,4) for v in win_tensor[f+1, k].tolist()]}")
else:
    print("\nNo large jumps found.")

print("=" * 60)











