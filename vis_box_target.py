"""
Minimal box (4_mice 18-keypt) sanity check:
  load val set -> pick a few rows -> prepare_batch -> render target_pose.
"""

from __future__ import annotations

import numpy as np
import torch

import befound
from befound.data.train_utils import prepare_batch
from befound.params import read as params_read
from neuroposelib import read as nplib_read
from neuroposelib import vis


# ============================================================================
# config
# ============================================================================
RESULTS_PATH = "/hpc/group/tdunn/joshwu/foundation/"
ANALYSIS_KEY = "ci_vae/2d_td"
LOAD_EPOCH   = 400
SAVE_ROOT    = "./"
N_VIS        = 100
DEVICE       = "cuda"
RNG_SEED     = 0


# ============================================================================
# 1) load val loader (model is built but unused)
# ============================================================================
config = params_read.config_load_only(
    f"{RESULTS_PATH.rstrip('/')}/{ANALYSIS_KEY}/model_config.yaml",
)
config["out_path"] = f"{RESULTS_PATH.rstrip('/')}/{ANALYSIS_KEY}/"
config["model"]["load_model"]  = config["out_path"]
config["model"]["start_epoch"] = LOAD_EPOCH
config["data"]["data_path"] = "/hpc/group/tdunn/joshwu/wu_iclr25/"

connectivity = nplib_read.connectivity_config(
    config["data"]["data_path"] + "mouse_skeleton.yaml",
)

loader_dict, _model = befound.get.data_and_model(
    config,
    train_val_test=["val"],
    data_keys=["x6d", "root", "offsets", "target_pose"],
    shuffle=[False],
    use_default_offsets=[True],
)
ds = loader_dict["val"].dataset

offsets_sum    = ds.offsets_sum
window         = config["model"]["window"]
n_keypts       = ds.n_keypts            # 18
kinematic_tree = ds.kinematic_tree
is_2d          = config["model"]["is_2d"]

augment_dict_off = {
    "2d_td": None, "kpt_shuffle": None,
    "offset_noise": None, "single_ablation": None,
}





# ============================================================================
# 2) pick N_VIS rows
# ============================================================================
rng = np.random.default_rng(RNG_SEED)
N_total = ds.data["x6d"].shape[0]
sel = np.sort(rng.choice(N_total, size=N_VIS, replace=False))
sel_t = torch.as_tensor(sel, dtype=torch.long)
print(f"selected box indices: {sel.tolist()}")


# ============================================================================
# 3) build subset dict (mirroring MouseDataset.__getitem__) + prepare_batch
# ============================================================================
data_subset = {}
for k, v in ds.data.items():
    v_t = v if torch.is_tensor(v) else torch.as_tensor(v)
    if k == "offsets" and getattr(ds, "standard_offsets", False):
        # shared (K, 3) when standard_offsets is on -- do NOT index
        data_subset[k] = v_t
    else:
        data_subset[k] = v_t[sel_t]

data_subset = prepare_batch(
    data=data_subset,
    augment_dict=augment_dict_off,
    offsets_sum=offsets_sum,
    kinematic_tree=kinematic_tree,
    device=DEVICE,
    get_2d=is_2d,
)

target_pose = data_subset["target_pose"].detach().cpu().numpy()  # (N_VIS, W, K, 3)
root        = data_subset["root"].detach().cpu().numpy()         # (N_VIS, W, 3)
print(f"target_pose: {target_pose.shape}, root: {root.shape}")


print(offsets_sum)
# print(target_pose)
# print(root)

print("\nroot per-window-frame min/max (median over frames):")
root_per_frame_min = root.min(axis=0)  # (W, 3) -- 每个 frame，N_VIS 个 window 的最小值
root_per_frame_max = root.max(axis=0)  # (W, 3) -- 每个 frame，N_VIS 个 window 的最大值

for axis, name in enumerate(['x', 'y', 'z']):
    min_median = np.median(root_per_frame_min[:, axis])
    max_median = np.median(root_per_frame_max[:, axis])
    print(f"  {name}: [{min_median:>9.4f}, {max_median:>9.4f}]")


# ============================================================================
# 4) render: target_pose is root-centered + unit-skeleton scale
# ============================================================================
pose = target_pose + root[:, :, None, :]            # back to world frame
flat = pose.reshape(-1, n_keypts, 3) * offsets_sum  # back to original units
assert flat.shape[0] == window * N_VIS, (flat.shape, window, N_VIS)

vis.pose.arena3D(
    flat,
    connectivity,
    frames=[0],
    centered=False,
    fps=30,
    N_FRAMES=window * N_VIS,
    VID_NAME="box_target_only.mp4",
    SAVE_ROOT=SAVE_ROOT,
)

print("done.")