"""
Minimal MABe22 sanity-check:
  load -> slice a few windows -> prepare_batch_2d_bespoke (Path A) -> render target_pose.
"""

from __future__ import annotations
from types import SimpleNamespace

import numpy as np
import torch
import matplotlib.cm as cm

from befound.data.train_utils import prepare_batch_2d_bespoke
from befound.data.constants import OFFSETS_3D_MABE22_SUM
from befound.get.get import get_mabe22_data
from neuroposelib import vis


# ============================================================================
# config
# ============================================================================
MABE_PATH = "/hpc/home/yw789/tdunn/befound_code/bams/data/mabe/"
SAVE_ROOT = "./"
WINDOW    = 51
N_VIS     = 20
DEVICE    = "cuda"

# MABe22 单只 mouse 的 12-keypt skeleton
PLOT_MOUSE_START_END = [
    (0, 1), (1, 3), (3, 2), (2, 0),         # head
    (3, 6), (6, 9),                         # midline
    (9, 10), (10, 11),                      # tail
    (4, 5), (5, 8), (8, 9), (9, 7), (7, 4), # legs
]
MABE_REINDEX = [6, 3, 0, 1, 2, 4, 5, 7, 8, 9, 10, 11]

inv = [0] * len(MABE_REINDEX)
for new, old in enumerate(MABE_REINDEX):
    inv[old] = new

PLOT_MOUSE_START_END = [(inv[a], inv[b]) for a, b in PLOT_MOUSE_START_END]

# ============================================================================
# 1) load MABe22 (reindexed to befound 顺序)
# ============================================================================
kp, _batch = get_mabe22_data(MABE_PATH, split="submission", reindex=True)
print(f"mabe submission keypoints: {kp.shape}")   # (N_traj, T, K=12, 2)


# ============================================================================
# 2) 切 N_VIS 个非重叠 window —— 取第 0 条 trajectory，沿时间均匀采样
# ============================================================================
N_TRAJ = kp.shape[0]
traj_indices = np.random.choice(N_TRAJ, size=N_VIS, replace=False)

half = WINDOW // 2
sampled_windows = []

for ti in traj_indices:
    traj = kp[ti]          # (T, K, 2)
    T = traj.shape[0]
    center = np.random.randint(half, T - half)
    sampled_windows.append(traj[center - half : center + half + 1])

windows = np.stack(sampled_windows, axis=0)   # (N_VIS, WINDOW, K, 2)
print(f"sampled traj indices: {traj_indices}")
print(f"windows: {windows.shape}")

# ============================================================================
# 3) Path A: 原始 2D pose -> midfwd 对齐 + 归一化
# ============================================================================
# 注意：bespoke 走 Path A 需要的就是原始 keypoints；不要预先算 x2d。
batch = {
    "pose": torch.from_numpy(windows).float(),   # (N_VIS, W, K, 2)
}

batch = prepare_batch_2d_bespoke(
    data=batch,
    offsets_sum=float(OFFSETS_3D_MABE22_SUM),
    device=DEVICE,
    augment_dict={"2d_td": None, "kpt_shuffle": None,
                   "offset_noise": None, "single_ablation": None},
)

target_pose = batch["target_pose"].detach().cpu().numpy()   # (N_VIS, W, K, 2)
root        = batch["root"].detach().cpu().numpy()          # (N_VIS, W, 3) 或 (N_VIS, W, 2)
print(f"target_pose: {target_pose.shape}, root: {root.shape}")

target_pose = target_pose + root[..., None, :2]
# print(root)

print("\nroot per-window-frame min/max (median over frames):")
root_per_frame_min = root.min(axis=0)  # (W, 3) -- 每个 frame，N_VIS 个 window 的最小值
root_per_frame_max = root.max(axis=0)  # (W, 3) -- 每个 frame，N_VIS 个 window 的最大值

for axis, name in enumerate(['x', 'y']):
    min_median = np.median(root_per_frame_min[:, axis])
    max_median = np.median(root_per_frame_max[:, axis])
    print(f"  {name}: [{min_median:>9.4f}, {max_median:>9.4f}]")

# ============================================================================
# 4) 渲染 target pose —— 2D 补一维抖动塞进 arena3D
# ============================================================================
pose3d = np.zeros((*target_pose.shape[:-1], 3), dtype=np.float32)
pose3d[..., :2] = np.nan_to_num(target_pose, nan=0.0)
pose3d[..., 2]  = np.random.uniform(-1e-3, 1e-3, size=pose3d.shape[:-1]).astype(np.float32)

K = pose3d.shape[2]

flat = pose3d.reshape(-1, K, 3) * float(OFFSETS_3D_MABE22_SUM)
assert flat.shape[0] == WINDOW * N_VIS, (flat.shape, WINDOW, N_VIS)

links = np.array(PLOT_MOUSE_START_END, dtype=int)
mabe_connectivity = SimpleNamespace(
    links=links,
    colors=cm.tab20(np.linspace(0, 1, len(links))),
    keypt_colors=cm.tab20(np.linspace(0, 1, K)),
)

vis.pose.arena3D(
    flat,
    mabe_connectivity,
    frames=[0],
    centered=False,
    fps=30,
    N_FRAMES=WINDOW * N_VIS,
    VID_NAME="mabe_target_only.mp4",
    SAVE_ROOT=SAVE_ROOT,
)

print("done.")