"""
Visualise 2D->3D lifting via joint latent space, on MABe22 mouse triplets.

Pipeline
--------
1. Box val set (4_mice, 18-keypt 3D), TWO-PASS to bound CPU RAM:
     pass A -- iterate the loader, collect ONLY mu (small)
     pass B -- after NN search, take the N_VIS hit indices and run a single
               forward through the model on just those rows; pull x6d,
               root, target_pose for those rows only

2. MABe22 submission set (12-keypt 2D)
   -> get_mabe22_data (split="submission", reindexed)
   -> prepare_batch_2d_bespoke (path A: raw pose, midfwd-aligned, z=0)
   -> per-window {mu, x6d, root}, sliding-window with stride=WINDOW so we
      get one non-overlapping tile per centre frame (M = ceil(T/stride))

3. Pick N_VIS MABe windows, find their 1-NN in box-mu space.

4. For each pair render three videos (concatenated along time):
       mabe_lifted.mp4  -- MABe x6d  -> fwd_kin -> 3D pose
       box_recon.mp4    -- box  x6d  -> fwd_kin -> 3D pose (reconstruction)
       box_target.mp4   -- box  target_pose (3D ground truth)

   All three are rendered with root_pos=0 so the comparison is pose-shape
   only, in the same skeleton space (18-keypt 4_mice).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from sklearn.neighbors import NearestNeighbors
from torch.utils.data import DataLoader

import befound
from befound.data import fwd_kin_cont6d_torch
from befound.data.train_utils import prepare_batch
from befound.params import read as params_read

from neuroposelib import read as nplib_read
from neuroposelib import vis

from befound.data.test_utils import (
    encode_decode_all,
    encode_decode_all_trajectories,
)
from befound.get.get import get_mabe22_data
from befound.data.constants import OFFSETS_3D_MABE22_SUM

from types import SimpleNamespace
import numpy as np
import matplotlib.cm as cm

# ============================================================================
# config
# ============================================================================

RESULTS_PATH = "/hpc/group/tdunn/joshwu/foundation/"
MABE_PATH    = "/hpc/home/yw789/tdunn/befound_code/bams/data/mabe/"
ANALYSIS_KEY = "ci_vae/2d_td"
LOAD_EPOCH   = 400

N_VIS         = 5      # how many MABe windows to lift + render
STRIDE_MABE   = 51     # non-overlapping windows over each MABe trajectory
BATCH_SIZE    = 512
DEVICE        = "cuda"
SAVE_ROOT     = "./"
RNG_SEED      = 0
ANALYSIS_TAG = ANALYSIS_KEY.replace("/", "_")

# ============================================================================
# 0) load model + box val loader
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

loader_dict, model = befound.get.data_and_model(
    config,
    train_val_test=["val"],
    data_keys=["x6d", "root", "offsets", "target_pose"],
    shuffle=[False],                 # NN alignment requires deterministic order
    use_default_offsets=[True],
)
loader = loader_dict["val"]
ds     = loader.dataset

offsets_sum     = ds.offsets_sum
window          = config["model"]["window"]
n_keypts        = ds.n_keypts            # 18 (4_mice training skeleton)
kinematic_tree  = ds.kinematic_tree
is_2d           = config["model"]["is_2d"]
default_offsets = ds.data["offsets"]     # (K, 3) shared offsets

augment_dict_off = {
    "2d_td": None, "kpt_shuffle": None,
    "offset_noise": None, "single_ablation": None,
}

model.eval()

# Rebuild the loader with num_workers=0. The default loader uses
# num_workers=5 (see get.get_mouse_data); on fork the workers do
# copy-on-write which can multiply ds.data's RAM footprint by ~5x and
# OOM the node before iteration even starts. We don't need workers here
# because pass A is dominated by GPU compute, not disk I/O.
loader = DataLoader(
    ds, batch_size=loader.batch_size, shuffle=False,
    num_workers=0, pin_memory=False,
)


# ============================================================================
# 1A) FIRST PASS: encode the whole box val set, collect ONLY mu
# ============================================================================

box_pass_a = encode_decode_all(
    model=model, loader=loader,
    augment_dict=augment_dict_off,
    offsets_sum=offsets_sum,
    kinematic_tree=kinematic_tree,
    get_2d=is_2d,
    keys={"mu"},                     # x6d / root deferred to pass B
    to_cpu=True,
)
mu_box = box_pass_a["mu"]            # (N_box, d_latent)
print(f"box pass A: mu {mu_box.shape}")


# ============================================================================
# 2) encode + decode MABe22 submission
# ============================================================================

# load + reindex + filter to submission split in one call
kp_sub, _batch_sub = get_mabe22_data(MABE_PATH, split="submission", reindex=True)
print(f"mabe submission: {kp_sub.shape}")

# Use the MABe22 unit-skeleton scale at inference time (~242.1) so the
# encoder sees its training-time magnitude.
mabe = encode_decode_all_trajectories(
    model=model, keypoints=kp_sub,
    offsets_sum=float(OFFSETS_3D_MABE22_SUM),
    window=window, batch_size=BATCH_SIZE, stride=STRIDE_MABE,
    device=DEVICE, keys={"mu", "x6d", "root"},
)
mu_mabe   = mabe["mu"]               # (N_mabe, M, d)
x6d_mabe  = mabe["x6d"]              # (N_mabe, M, window, n_keypts, 6)
root_mabe = mabe["root"]             # (N_mabe, M, window, 3)
mabe_target_pose = mabe["target_pose"]  # (N_mabe, M, window, n_keypts, 2)
N_mabe, M = mu_mabe.shape[:2]
print(f"mabe: mu {mu_mabe.shape}, x6d {x6d_mabe.shape}, root {root_mabe.shape}")


# ============================================================================
# 3) pick N_VIS MABe windows; find their 1-NN in box-mu space
# ============================================================================

mu_mabe_flat = mu_mabe.reshape(N_mabe * M, -1)         # (N_mabe*M, d)
rng  = np.random.default_rng(RNG_SEED)
sel  = rng.choice(mu_mabe_flat.shape[0], size=N_VIS, replace=False)
sel  = np.sort(sel)                                    # cosmetic
sel_n, sel_m = np.divmod(sel, M)                       # back to (n, m)

print(f"selected MABe (n, m) pairs: {list(zip(sel_n.tolist(), sel_m.tolist()))}")

nbrs = NearestNeighbors(n_neighbors=1, metric="euclidean", n_jobs=-1).fit(mu_box)
dists, nn_idx = nbrs.kneighbors(mu_mabe_flat[sel])     # (N_VIS, 1)
nn_idx = nn_idx.ravel()                                # (N_VIS,)
print(f"NN distances: {dists.ravel()}")
print(f"NN box indices: {nn_idx}")


# ============================================================================
# 1B) SECOND PASS: rebuild the N_VIS NN rows of the box dataset and decode
# ============================================================================
# We build a small batch dict by indexing each entry of `ds.data` with
# nn_idx (except `offsets`, which is shared (K, 3) when standard_offsets
# is on -- see MouseDataset.__getitem__). This mirrors the dataset's own
# batching logic without going through the DataLoader.

nn_idx_t = torch.as_tensor(nn_idx, dtype=torch.long)

data_subset = {}
for k, v in ds.data.items():
    if k == "offsets" and getattr(ds, "standard_offsets", False):
        data_subset[k] = v if torch.is_tensor(v) else torch.as_tensor(v)
    else:
        v_t = v if torch.is_tensor(v) else torch.as_tensor(v)
        data_subset[k] = v_t[nn_idx_t]

# Run prepare_batch + a single forward on just N_VIS rows.
data_subset = prepare_batch(
    data=data_subset,
    augment_dict=augment_dict_off,
    offsets_sum=offsets_sum,
    kinematic_tree=kinematic_tree,
    device=DEVICE,
    get_2d=is_2d,
)
with torch.no_grad():
    data_o_subset = model(data_subset)

x6d_box_nn      = data_o_subset["x6d"].detach().cpu().numpy()        # (N_VIS, W, K, 6)
x6d_root_nn      = data_o_subset["root"].detach().cpu().numpy()        # (N_VIS, W, K, 6)
target_pose_nn  = data_subset["target_pose"].detach().cpu().numpy()  # (N_VIS, W, K, 3)
target_pose_root_nn  = data_subset["root"].detach().cpu().numpy()  # (N_VIS, W, K, 3)
print(f"box pass B: x6d {x6d_box_nn.shape}, target_pose {target_pose_nn.shape}")


# ============================================================================
# 4) decode 6d -> 3D pose for each row, then render
# ============================================================================

# def fwd_kin_at_origin(x6d: np.ndarray, root_t: np.ndarray) -> np.ndarray:
#     """
#     x6d: (window, n_keypts, 6) -> 3D pose of shape (window, n_keypts, 3),
#     rendered with root translation set to zero so we compare pose shape only.
#     Operates on numpy via a single torch round-trip.
#     """
#     x6d_t = torch.from_numpy(x6d).reshape(-1, n_keypts, 6).to(DEVICE).float()
#     if torch.is_tensor(default_offsets):
#         offsets_t = default_offsets.to(DEVICE).float() / offsets_sum
#     else:
#         offsets_t = torch.from_numpy(default_offsets).to(DEVICE).float() / offsets_sum
#     # root_t = torch.zeros(x6d_t.shape[0], 3, device=DEVICE)
#     pose_t = fwd_kin_cont6d_torch(
#         x6d_t, kinematic_tree, offsets_t,
#         root_pos=root_t, do_root_R=True, eps=1e-8,
#     )
#     return pose_t.reshape(window, n_keypts, 3).detach().cpu().numpy()

def fwd_kin_at_origin(x6d, root):           # 名字叫 origin 但其实带 root，留给你看
    x6d_t = torch.from_numpy(x6d).reshape(-1, n_keypts, 6).to(DEVICE).float()
    if torch.is_tensor(default_offsets):
        offsets_t = default_offsets.to(DEVICE).float() / offsets_sum
    else:
        offsets_t = torch.from_numpy(default_offsets).to(DEVICE).float() / offsets_sum
    root_t = torch.as_tensor(root).reshape(-1, 3).to(DEVICE).float()
    pose_t = fwd_kin_cont6d_torch(
        x6d_t, kinematic_tree, offsets_t,
        root_pos=root_t, do_root_R=True, eps=1e-8,
    )
    return pose_t.reshape(window, n_keypts, 3).detach().cpu().numpy()


# Stack the N_VIS windows along time axis: each row is (window, n_keypts, 3).
mabe_lifted = np.stack(
    [fwd_kin_at_origin(x6d_mabe[n, m], root_mabe[n, m]) for n, m in zip(sel_n, sel_m)], axis=0,
)                                                  # (N_VIS, window, n_keypts, 3)
box_recon   = np.stack(
    [fwd_kin_at_origin(x6d_box_nn[i], x6d_root_nn[i]) for i in range(N_VIS)], axis=0,
)                                                  # (N_VIS, window, n_keypts, 3)
# box_target  = target_pose_nn + target_pose_root_nn                       # (N_VIS, window, n_keypts, 3)
box_target = target_pose_nn + target_pose_root_nn[:, :, None, :]   # (N_VIS, window, n_keypts, 3)


# Convert from unit-skeleton scale back to original units before rendering.
# `target_pose` is built with normalized offsets too (see data.py:544), so
# all three need the same multiplier.
def render(pose, vid_name):
    flat = pose.reshape(-1, n_keypts, 3) * offsets_sum
    vis.pose.arena3D(
        flat,
        connectivity,
        frames=[0],
        centered=False,
        fps=30,
        N_FRAMES=window * N_VIS,
        VID_NAME=vid_name,
        SAVE_ROOT=SAVE_ROOT,
    )

render(mabe_lifted, f"{ANALYSIS_TAG}_mabe_lifted.mp4")
render(box_recon,   f"{ANALYSIS_TAG}_box_recon.mp4")
render(box_target,  f"{ANALYSIS_TAG}_box_target.mp4")

print("done. videos written to", Path(SAVE_ROOT).resolve())


PLOT_MOUSE_START_END = [(0, 1), (1, 3), (3, 2), (2, 0),        # head
                        (3, 6), (6, 9),                        # midline
                        (9, 10), (10, 11),                     # tail
                        (4, 5), (5, 8), (8, 9), (9, 7), (7, 4) # legs
                       ]

mabe_target_2d = np.stack(
    [mabe_target_pose[n, m] for n, m in zip(sel_n, sel_m)], axis=0,
)  # (N_VIS, window, K_mabe, 2)

K_mabe = mabe_target_2d.shape[-2]   # 用一个新变量，别覆盖 n_keypts

mabe_target_3d = np.zeros((*mabe_target_2d.shape[:-1], 3), dtype=np.float32)
mabe_target_3d[..., :2] = np.nan_to_num(mabe_target_2d, nan=0.0)
mabe_target_3d[..., 2] = np.random.uniform(
    -1e-3, 1e-3, size=mabe_target_3d.shape[:-1],
).astype(np.float32)

flat = mabe_target_3d.reshape(-1, K_mabe, 3) * float(OFFSETS_3D_MABE22_SUM)
assert flat.shape[0] == window * N_VIS, (flat.shape, window, N_VIS)

links = np.array(PLOT_MOUSE_START_END, dtype=int)
assert links.max() < K_mabe, f"links idx {links.max()} >= K_mabe {K_mabe}"
n_links = len(links)

mabe_connectivity = SimpleNamespace(
    links=links,
    colors=cm.tab20(np.linspace(0, 1, n_links)),
    keypt_colors=cm.tab20(np.linspace(0, 1, K_mabe)),
)

vis.pose.arena3D(
    flat,
    mabe_connectivity,                  # ← 用 mabe 的，不是 box 的 connectivity
    frames=[0],
    centered=False,
    fps=30,
    N_FRAMES=window * N_VIS,
    VID_NAME=f"{ANALYSIS_TAG}_mabe_target.mp4",
    SAVE_ROOT=SAVE_ROOT,
)


# import pdb
# pdb.set_trace()


# """
# Visualise 2D->3D lifting via joint latent space, on MABe22 mouse triplets.

# Pipeline
# --------
# 1. Box val set (4_mice, 18-keypt 3D)
#    -> prepare_batch (3D path, get_2d=False)
#    -> model.encode + model.decode
#    -> per-window {mu, x6d, root} for the whole val loader
#    -> plus target_pose pulled directly from loader.dataset.data
#       (alignable because shuffle=False)

# 2. MABe22 submission set (12-keypt 2D)
#    -> load_mice_triplet + ABE_REINDEX
#    -> prepare_batch_2d_bespoke (path A: raw pose, midfwd-aligned, z=0)
#    -> model.encode + model.decode (CI encoder swallows K=12;
#       decoder still emits 18-keypt x6d/root because the head is fixed-channel)
#    -> per-window {mu, x6d, root}, sliding-window with `stride=WINDOW` so we
#       get one non-overlapping tile per centre frame (M = ceil(T/stride))

# 3. Pick N_VIS MABe windows, find their 1-NN in box-mu space.

# 4. For each pair render three videos (concatenated along time):
#        VID_mabe_lifted.mp4  -- MABe x6d  -> fwd_kin -> 3D pose
#        VID_box_recon.mp4    -- box  x6d  -> fwd_kin -> 3D pose (reconstruction)
#        VID_box_target.mp4   -- box  target_pose (3D ground truth)

#    All three are rendered with root_pos=0 so the comparison is pose-shape
#    only, in the same skeleton space (18-keypt 4_mice).

# """

# from __future__ import annotations

# from pathlib import Path

# import numpy as np
# import torch
# from sklearn.neighbors import NearestNeighbors

# import befound
# from befound.data import fwd_kin_cont6d_torch
# from befound.data.train_utils import prepare_batch
# from befound.params import read as params_read

# from neuroposelib import read as nplib_read
# from neuroposelib import vis

# from befound.data.test_utils import (
#     encode_decode_all,
#     encode_decode_all_trajectories,
# )
# from befound.get.get import get_mabe22_data
# from befound.data.constants import OFFSETS_3D_MABE22_SUM


# # ============================================================================
# # config
# # ============================================================================

# RESULTS_PATH = "/hpc/group/tdunn/joshwu/foundation/"
# MABE_PATH    = "/hpc/home/yw789/tdunn/befound_code/bams/data/mabe/"
# ANALYSIS_KEY = "ci_vae/2d_td"
# LOAD_EPOCH   = 400

# N_VIS         = 5      # how many MABe windows to lift + render
# STRIDE_MABE   = 51     # non-overlapping windows over each MABe trajectory
# BATCH_SIZE    = 512
# DEVICE        = "cuda"
# SAVE_ROOT     = "./"
# RNG_SEED      = 0


# # ============================================================================
# # 0) load model + box val loader
# # ============================================================================

# config = params_read.config_load_only(f"{RESULTS_PATH.rstrip('/')}/{ANALYSIS_KEY}/model_config.yaml")
# config["out_path"] = f"{RESULTS_PATH.rstrip('/')}/{ANALYSIS_KEY}/"
# config["model"]["load_model"]  = config["out_path"]
# config["model"]["start_epoch"] = LOAD_EPOCH
# config["data"]["data_path"] = "/hpc/group/tdunn/joshwu/wu_iclr25/"

# connectivity = nplib_read.connectivity_config(
#     config["data"]["data_path"] + "mouse_skeleton.yaml"
# )

# loader_dict, model = befound.get.data_and_model(
#     config,
#     train_val_test=["val"],
#     data_keys=["x6d", "root", "offsets", "target_pose"],
#     shuffle=[False],                # NN alignment requires deterministic order
#     use_default_offsets=[True],
# )
# loader = loader_dict["val"]
# ds = loader.dataset

# offsets_sum     = ds.offsets_sum
# window          = config["model"]["window"]
# n_keypts        = ds.n_keypts            # 18 (4_mice training skeleton)
# kinematic_tree  = ds.kinematic_tree
# is_2d           = config["model"]["is_2d"]
# default_offsets = ds.data["offsets"]     # (n_keypts, 3), shared offsets

# augment_dict_off = {
#     "2d_td": None, "kpt_shuffle": None,
#     "offset_noise": None, "single_ablation": None,
# }

# model.eval()


# # ============================================================================
# # 1) encode + decode the whole box val set
# # ============================================================================

# box = encode_decode_all(
#     model=model, loader=loader,
#     augment_dict=augment_dict_off,
#     offsets_sum=offsets_sum,
#     kinematic_tree=kinematic_tree,
#     get_2d=is_2d,
#     keys={"mu", "x6d", "root"},
#     to_cpu=True,
# )
# mu_box   = box["mu"]      # (N_box, d_latent)
# x6d_box  = box["x6d"]     # (N_box, window, n_keypts, 6)
# root_box = box["root"]    # (N_box, window, 3)

# # target_pose is an input field, not a model output -- pull it directly.
# # loader was built with shuffle=False so iteration order matches the
# # dataset's underlying array order, which is what we use.
# target_pose_box = ds.data["target_pose"]
# if torch.is_tensor(target_pose_box):
#     target_pose_box = target_pose_box.detach().cpu().numpy()
# target_pose_box = np.asarray(target_pose_box, dtype=np.float32)
# assert target_pose_box.shape[0] == mu_box.shape[0], (
#     f"target_pose ({target_pose_box.shape[0]}) and mu_box ({mu_box.shape[0]}) "
#     "lengths disagree -- did the loader shuffle?"
# )
# print(f"box: mu {mu_box.shape}, x6d {x6d_box.shape}, root {root_box.shape}, "
#       f"target {target_pose_box.shape}")


# # ============================================================================
# # 2) encode + decode MABe22 submission
# # ============================================================================

# # load + reindex + filter to submission split in one call
# kp_sub, _batch_sub = get_mabe22_data(MABE_PATH, split="submission", reindex=True)
# print(f"mabe submission: {kp_sub.shape}")

# # Use the MABe22 unit-skeleton scale at inference time (242.1) so the
# # encoder sees its training-time magnitude.
# mabe = encode_decode_all_trajectories(
#     model=model, keypoints=kp_sub,
#     offsets_sum=float(OFFSETS_3D_MABE22_SUM),
#     window=window, batch_size=BATCH_SIZE, stride=STRIDE_MABE,
#     device=DEVICE, keys={"mu", "x6d", "root"},
# )
# mu_mabe   = mabe["mu"]      # (N_mabe, M, d)
# x6d_mabe  = mabe["x6d"]     # (N_mabe, M, window, n_keypts, 6)
# root_mabe = mabe["root"]    # (N_mabe, M, window, 3)
# N_mabe, M = mu_mabe.shape[:2]
# print(f"mabe: mu {mu_mabe.shape}, x6d {x6d_mabe.shape}, root {root_mabe.shape}")


# # ============================================================================
# # 3) pick N_VIS MABe windows; find their 1-NN in box-mu space
# # ============================================================================

# mu_mabe_flat = mu_mabe.reshape(N_mabe * M, -1)         # (N_mabe*M, d)
# rng  = np.random.default_rng(RNG_SEED)
# sel  = rng.choice(mu_mabe_flat.shape[0], size=N_VIS, replace=False)
# sel  = np.sort(sel)                                    # cosmetic
# sel_n, sel_m = np.divmod(sel, M)                       # back to (n, m)

# print(f"selected MABe (n, m) pairs: {list(zip(sel_n.tolist(), sel_m.tolist()))}")

# nbrs = NearestNeighbors(n_neighbors=1, metric="euclidean", n_jobs=-1).fit(mu_box)
# dists, nn_idx = nbrs.kneighbors(mu_mabe_flat[sel])     # (N_VIS, 1)
# nn_idx = nn_idx.ravel()                                 # (N_VIS,)
# print(f"NN distances: {dists.ravel()}")
# print(f"NN box indices: {nn_idx}")


# # ============================================================================
# # 4) decode 6d -> 3D pose for each row, then render
# # ============================================================================

# def fwd_kin_at_origin(x6d: np.ndarray) -> np.ndarray:
#     """
#     x6d: (..., window, n_keypts, 6) -> 3D pose of shape (window, n_keypts, 3),
#     rendered with root translation set to zero so we compare pose shape only.
#     Operates on numpy via a single torch round-trip.
#     """
#     x6d_t = torch.from_numpy(x6d).reshape(-1, n_keypts, 6).to(DEVICE).float()
#     offsets_t = torch.from_numpy(default_offsets).to(DEVICE).float() / offsets_sum
#     root_t = torch.zeros(x6d_t.shape[0], 3, device=DEVICE)
#     pose_t = fwd_kin_cont6d_torch(
#         x6d_t, kinematic_tree, offsets_t,
#         root_pos=root_t, do_root_R=True, eps=1e-8,
#     )
#     return pose_t.reshape(window, n_keypts, 3).detach().cpu().numpy()


# # Stack the N_VIS windows along time axis: each row is (window, n_keypts, 3).
# mabe_lifted = np.stack(
#     [fwd_kin_at_origin(x6d_mabe[n, m]) for n, m in zip(sel_n, sel_m)], axis=0,
# )                                                  # (N_VIS, window, n_keypts, 3)
# box_recon   = np.stack(
#     [fwd_kin_at_origin(x6d_box[b]) for b in nn_idx], axis=0,
# )                                                  # (N_VIS, window, n_keypts, 3)
# box_target  = target_pose_box[nn_idx]              # (N_VIS, window, n_keypts, 3)

# # Convert from unit-skeleton scale back to original units before rendering.
# # `target_pose` is built with normalized offsets too (see data.py:544), so
# # all three need the same multiplier.
# def render(pose, vid_name):
#     flat = pose.reshape(-1, n_keypts, 3) * offsets_sum
#     vis.pose.arena3D(
#         flat,
#         connectivity,
#         frames=[0],
#         centered=False,
#         fps=30,
#         N_FRAMES=window * N_VIS,
#         VID_NAME=vid_name,
#         SAVE_ROOT=SAVE_ROOT,
#     )

# render(mabe_lifted, "mabe_lifted.mp4")
# render(box_recon,   "box_recon.mp4")
# render(box_target,  "box_target.mp4")

# print("done. videos written to", Path(SAVE_ROOT).resolve())

# import pdb
# pdb.set_trace()



#################################################################################


# import befound
# from neuroposelib import read
# import torch
# from befound.data.train_utils import prepare_batch
# import numpy as np
# from scrubvae.data.dataset import fwd_kin_cont6d_torch
# from neuroposelib import vis
# from befound.get.get import get_mabe22_data

# results_path = "/hpc/group/tdunn/joshwu/foundation/"
# mabe_path = "/hpc/group/tdunn/joshwu/foundation/mabe22_data/"
# analysis_key = "ci_vae/2d_td"

# config = read.config_load_only("{}/{}/model_config.yaml".format(results_path, analysis_key))
# config["out_path"] = f"{results_path.rstrip('/')}/{analysis_key}/"
# config["model"]["load_model"] = config["out_path"]
# config["model"]["start_epoch"] = 400

# connectivity = read.connectivity_config(
#     config["data"]["data_path"] + "mouse_skeleton.yaml"
# )

# loader_dict, model = befound.get.data_and_model(
#     config,
#     train_val_test=["val"],
#     data_keys=["x6d", "root", "offsets", "target_pose"],
#     shuffle=[True, False],
#     use_default_offsets=[True, False],
# )
# loader = loader_dict["val"]
# offsets_sum = loader.dataset.offsets_sum
# window = config["model"]["window"]
# n_keypts = loader.dataset.n_keypts
# kinematic_tree = loader.dataset.kinematic_tree

# augment_dict = {
#     "2d_td": None,
#     "kpt_shuffle": None,
#     "offset_noise": None,
#     "single_ablation": None,
# }

# model.eval()
# # sample_inds = [10000, 100000, 253405]
# z = []
# # data = loader.dataset[sample_inds]
# data = loader.dataset
# data_box = prepare_batch(
#     data=data,
#     augment_dict=augment_dict,
#     offsets_sum=offsets_sum,
#     kinematic_tree=kinematic_tree,
#     device="cuda",
#     get_2d=config["model"]["is_2d"],
# )
# data_o_box = model(data_box) # all of box data embedded

# ######
# # get_mabe data and mu
# ######

# # lantent_mabe = np.load("/hpc/home/yw789/tdunn/befound_code/befound/submissions_analysis/2dtd_submission_nseq_3_T_D.npy", allow_pickle=True).item()
# # print("MABE22 latents shape: {}".format(lantent_mabe.shape))

# kp, batch = get_mabe22_data(mabe_path, split="ssubmission", reindex=True)

# z_sub = encode_all_trajectories(
#             model=model, keypoints=kp_sub, offsets_sum=offsets_sum,
#             window=WINDOW, batch_size=batch_size, device=device,
#         )


# # data_o_mabe = model(data_mabe)
# # data_o_mabe["mu"] # latents of MABE




# mabe_nn = nearest_neighbor(data_o_mabe["mu"], data_o_box["mu"]) # gives you value in data_o_box in which data_o_box["mu"] is closest to data_o_mabe["mu"]

# pose = fwd_kin_cont6d_torch(
#     mabe_nn["x6d"],
#     kinematic_tree,
#     offsets.reshape((-1, n_keypts, 3)),
#     root_pos=mabe_nn["root"].reshape(-1, 3),
#     do_root_R=True,
# ).reshape(-1, window, n_keypts, 3)

# dissimilarity = scipy.spatial.procrustes_by_batch(
#     pose[..., keypoints_in_mabe, :2].reshape(data_len, window*n_keypts, 2), 
#     data_mabe["x2d"][..., keypoints_in_box, :].reshape(data_len, window*n_keypts, 2), dim=-1).mean()
# assert (dissimilarity_with_lifting < dissimilarity_without_lifting)

# pose = fwd_kin_cont6d_torch(
#     data_o["x6d"].reshape((-1, n_keypts, 6)),
#     kinematic_tree,
#     offsets.reshape((-1, n_keypts, 3)),
#     root_pos=data_o["root"].reshape(-1, 3),
#     do_root_R=True,
# ).reshape(-1, window, n_keypts, 3)














# pose = np.zeros((*data_o["x2d"].shape[:-1], 3)) + 0.01
# pose[..., :2] = data_o["x2d"].cpu().detach().numpy()
# vis.pose.arena3D(
#     pose.reshape(-1, n_keypts, 3) * offsets_sum,
#     connectivity,
#     frames=[0],
#     centered=False,
#     fps=30,
#     N_FRAMES=window * len(sample_inds),
#     VID_NAME=f"test_2d_o.mp4",
#     SAVE_ROOT="./",
# )

# pose = np.zeros((*data["x2d"].shape[:-1], 3)) + 0.01
# pose[..., :2] = data["x2d"].cpu().detach().numpy()
# vis.pose.arena3D(
#     pose.reshape(-1, n_keypts, 3) * offsets_sum,
#     connectivity,
#     frames=[0],
#     centered=False,
#     fps=30,
#     N_FRAMES=window * len(sample_inds),
#     VID_NAME=f"test_2d.mp4",
#     SAVE_ROOT="./",
# )

# pose = np.zeros((*data["target_pose"].shape[:-1], 3)) + 0.01
# pose[..., :2] = data["target_pose"][..., :2].cpu().detach().numpy()
# vis.pose.arena3D(
#     pose.reshape(-1, n_keypts, 3) * offsets_sum,
#     connectivity,
#     frames=[0],
#     centered=False,
#     fps=30,
#     N_FRAMES=window * len(sample_inds),
#     VID_NAME=f"test_2d_target.mp4",
#     SAVE_ROOT="./",
# )

# import pdb

# pdb.set_trace()
