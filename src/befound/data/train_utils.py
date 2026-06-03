from befound.data import fwd_kin_cont6d_torch
from befound.data import quaternion as qtn
from typing import Dict, List, Optional
import numpy as np
import torch


def prepare_batch(
    data: Dict,
    augment_dict: Dict,
    offsets_sum: float,
    kinematic_tree: List[int],
    device: str = "cuda",
    get_2d: bool = False,
):
    if "offsets" in data.keys():
        is_default_offsets = len(data["offsets"].shape) == 3

    for k, v in data.items():
        if k == "offsets" and is_default_offsets:
            data[k] = v[0].to(device)
        else:
            data[k] = v.to(device)

    offsets = data["offsets"].clone()
    if augment_dict["offset_noise"]:
        if is_default_offsets:
            offsets = offsets.expand(
                data["x6d"].shape[:-2]
                + (
                    -1,
                    -1,
                )
            ).clone()

        eps = (augment_dict["offset_noise"] * torch.randn_like(offsets)).mul(offsets)
        offsets = offsets.add_(eps)

    x3d = get_x3d_from_data(
        data["x6d"], offsets, data["root"], offsets_sum, kinematic_tree
    )

    if augment_dict["single_ablation"]:
        mask = torch.rand(x3d.shape[0]) < augment_dict["single_ablation"]
        keypt_to_ablate = torch.randint(0, x3d.shape[-2], (mask.sum(),), device=device)
        x3d[mask][..., keypt_to_ablate, :] = 0

    if augment_dict["2d_td"]:
        mask = torch.rand(x3d.shape[0]) < augment_dict["2d_td"]
        x3d[mask, :, :, 2] = 0

    if augment_dict["kpt_shuffle"]:
        B, W, N_K, D = x3d.shape
        rand_ind = torch.rand(B, N_K).to(device).argsort(dim=-1)
        idx = rand_ind[:, None, :, None].expand(-1, W, -1, D)
        x3d = torch.gather(x3d, dim=2, index=idx)

    # [yw] temporal masking:
    # mode 1: mask first 25 frames, predict the next 26
    if augment_dict.get("temporal_mask_past"):
        mask = torch.rand(x3d.shape[0]) < augment_dict["temporal_mask_past"]
        if mask.any():
            x3d[mask, :25] = 0

    # mode 2: mask first 26 frames, predict the last 25
    if augment_dict.get("temporal_mask_future"):
        mask = torch.rand(x3d.shape[0]) < augment_dict["temporal_mask_future"]
        if mask.any():
            x3d[mask, :26] = 0

    # mode 3: mask up to max_len consecutive frames in the interior
    if augment_dict.get("temporal_mask_random"):
        prob, max_len = augment_dict["temporal_mask_random"]
        max_len = int(max_len)
        B, W = x3d.shape[:2]
        apply = torch.rand(B, device=x3d.device) < prob # (B,)
        lengths = torch.randint(1, max_len + 1, (B,), device=x3d.device)
        starts = torch.randint(1, W - max_len, (B,), device=x3d.device)
        t = torch.arange(W, device=x3d.device)[None, :] # (1, W)
        frame_mask = apply[:, None] & (t >= starts[:, None]) & (t < (starts + lengths)[:, None]) # (B, W)
        x3d = x3d.masked_fill(frame_mask[:, :, None, None], 0.0)

    if get_2d:
        data["x2d"] = x3d[...,:2].clone()
    else:
        data["x3d"] = x3d.clone()
    return data

# def prepare_batch_2d_bespoke(
#     data: Dict,
#     augment_dict: Dict,
#     offsets_sum: float,
#     device: str = "cuda",
# ):
#     if "x2d" not in data and "pose" in data:

#         pose_window = data["pose"]
#         N, window, N_KEYPTS, _ = pose_window.shape
#         half_w = window // 2
#         pose = pose_window.astype(np.float32)

#         root_xy = pose[:, half_w, 0, :2].copy()
#         pose_centered = pose.copy()
#         pose_centered[..., :2] -= root_xy[:, None, None, :]

#         forward = pose[:, half_w, 1, :] - pose[:, half_w, 0, :]
#         forward /= np.linalg.norm(forward, axis=-1, keepdims=True) + 1e-8
#         yaw = -np.arctan2(forward[:, 1], forward[:, 0])[..., None]

#         fwd_q = np.zeros((N, 4), dtype=np.float32)
#         fwd_q[:, [-1, 0]] = np.concatenate([np.sin(yaw / 2), np.cos(yaw / 2)], axis=-1)
#         fwd_q_w = np.repeat(fwd_q[:, None, :], window, axis=1)

#         fwd_q_joints = fwd_q_w[:, :, None, :].repeat(N_KEYPTS, axis=2)
#         pose_aligned = qtn.qrot_np(
#             fwd_q_joints.reshape(-1, 4), pose_centered.reshape(-1, 3),
#         ).reshape(N, window, N_KEYPTS, 3)

#         # Convert to torch tensor and move to device before augmentations
#         x3d = torch.from_numpy(pose_aligned).to(device) / offsets_sum

#         if augment_dict["single_ablation"]:
#             mask = torch.rand(x3d.shape[0]) < augment_dict["single_ablation"]
#             keypt_to_ablate = torch.randint(0, x3d.shape[-2], (mask.sum(),), device=device)
#             x3d[mask][..., keypt_to_ablate, :] = 0

#         if augment_dict["2d_td"]:
#             mask = torch.rand(x3d.shape[0]) < augment_dict["2d_td"]
#             x3d[mask, :, :, 2] = 0

#         if augment_dict["kpt_shuffle"]:
#             B, W, N_K, D = x3d.shape
#             rand_ind = torch.rand(B, N_K).to(device).argsort(dim=-1)
#             idx = rand_ind[:, None, :, None].expand(-1, W, -1, D)
#             x3d = torch.gather(x3d, dim=2, index=idx)

#         data["x2d"] = x3d[...,:2].clone()
    
#     else:
#         x2d = data["x2d"]

#          if augment_dict["single_ablation"]:
#             mask = torch.rand(x2d.shape[0]) < augment_dict["single_ablation"]
#             keypt_to_ablate = torch.randint(0, x2d.shape[-2], (mask.sum(),), device=device)
#             x2d[mask][..., keypt_to_ablate, :] = 0

#         # if augment_dict["2d_td"]:
#         #     mask = torch.rand(x3d.shape[0]) < augment_dict["2d_td"]
#         #     x3d[mask, :, :, 2] = 0

#         if augment_dict["kpt_shuffle"]:
#             B, W, N_K, D = x2d.shape
#             rand_ind = torch.rand(B, N_K).to(device).argsort(dim=-1)
#             idx = rand_ind[:, None, :, None].expand(-1, W, -1, D)
#             x2d = torch.gather(x2d, dim=2, index=idx)
        
#         data["x2d"] = x2d.clone()


#     return data

def prepare_batch_2d_bespoke(
    data: Dict,
    augment_dict: Optional[Dict],
    offsets_sum: float,
    device: str = "cuda",
):
    if "x2d" not in data and "pose" in data:
        # ---- path A: raw pose -> midfwd-aligned x3d -----------------
        pose = data["pose"]
        if torch.is_tensor(pose):
            pose = pose.detach().cpu().numpy()
        pose = np.asarray(pose, dtype=np.float32)

        if pose.shape[-1] == 2:
            pose = np.concatenate(
                [pose, np.zeros_like(pose[..., :1])], axis=-1
            )
        assert pose.shape[-1] == 3, (
            f"path A expects pose of shape (B, W, K, 2 or 3), got {pose.shape}"
        )

        # missing_kpt = (pose[..., 0] == 0) & (pose[..., 1] == 0)   # (N, W, K)
        # frame_invalid = missing_kpt.any(axis=-1)                  # (N, W)
        # missing_kpt = (pose[..., 0] == 0) & (pose[..., 1] == 0)   # (N, W, K)
        # kpt_valid_np = ~missing_kpt
        # loss_mask = torch.from_numpy(~kpt_valid_np).to(device)   # (N, W) bool

        N, window, N_KEYPTS, _ = pose.shape
        half_w = window // 2

        root_xy = pose[:, half_w, 0, :2].copy()
        pose_centered = pose.copy()
        pose_centered[..., :2] -= root_xy[:, None, None, :]

        forward = pose[:, half_w, 1, :2] - pose[:, half_w, 0, :2]
        forward /= np.linalg.norm(forward, axis=-1, keepdims=True) + 1e-8
        yaw = -np.arctan2(forward[:, 1], forward[:, 0])[..., None]

        fwd_q = np.zeros((N, 4), dtype=np.float32)
        fwd_q[:, [-1, 0]] = np.concatenate(
            [np.sin(yaw / 2), np.cos(yaw / 2)], axis=-1,
        )
        fwd_q_w = np.repeat(fwd_q[:, None, :], window, axis=1)
        fwd_q_joints = fwd_q_w[:, :, None, :].repeat(N_KEYPTS, axis=2)
        pose_aligned = qtn.qrot_np(
            fwd_q_joints.reshape(-1, 4),
            pose_centered.reshape(-1, 3),
        ).reshape(N, window, N_KEYPTS, 3)

        x3d = torch.from_numpy(pose_aligned).to(device) / offsets_sum
        x2d = x3d[..., :2].clone()

        data["target_pose"] = (x2d - x2d[:, :, 0:1]) * offsets_sum
        data["root"]        = x2d[:, :, 0] * offsets_sum
        # data["loss_mask"] = loss_mask

        if augment_dict["single_ablation"]:
            mask = torch.rand(x3d.shape[0], device=device) < augment_dict["single_ablation"]
            n_masked = int(mask.sum())
            if n_masked > 0:
                masked_idx = mask.nonzero(as_tuple=True)[0]                 # (n_masked,)
                keypt_to_ablate = torch.randint(
                    0, x3d.shape[-2], (n_masked,), device=device,
                )                                                           # (n_masked,)
                x3d[masked_idx, :, keypt_to_ablate, :] = 0
                x2d[masked_idx, :, keypt_to_ablate, :] = 0

        if augment_dict["2d_td"]:
            mask = torch.rand(x3d.shape[0], device=device) < augment_dict["2d_td"]
            x3d[mask, :, :, 2] = 0

        if augment_dict["kpt_shuffle"]:
            B, W, N_K, D3 = x3d.shape
            rand_ind = torch.rand(B, N_K, device=device).argsort(dim=-1)
            x3d = torch.gather(
                x3d, dim=2,
                index=rand_ind[:, None, :, None].expand(-1, W, -1, D3),
            )
            x2d = torch.gather(
                x2d, dim=2,
                index=rand_ind[:, None, :, None].expand(-1, W, -1, x2d.shape[-1]),
            )

        # [yw] temporal masking
        # mode 1: mask first 25 frames, predict the next 26
        if augment_dict.get("temporal_mask_past"):
            mask = torch.rand(x2d.shape[0], device=device) < augment_dict["temporal_mask_past"]
            if mask.any():
                x2d[mask, :25] = 0
                x3d[mask, :25] = 0

        # mode 2: mask first 26 frames, predict the last 25
        if augment_dict.get("temporal_mask_future"):
            mask = torch.rand(x2d.shape[0], device=device) < augment_dict["temporal_mask_future"]
            if mask.any():
                x2d[mask, :26] = 0
                x3d[mask, :26] = 0

        # mode 3: mask up to max_len consecutive frames in the interior
        if augment_dict.get("temporal_mask_random"):
            prob, max_len = augment_dict["temporal_mask_random"]
            max_len = int(max_len)
            B, W = x2d.shape[:2]
            apply = torch.rand(B, device=device) < prob                          # (B,)
            lengths = torch.randint(1, max_len + 1, (B,), device=device)
            starts = torch.randint(1, W - max_len, (B,), device=device)
            t = torch.arange(W, device=device)[None, :]                          # (1, W)
            frame_mask = apply[:, None] & (t >= starts[:, None]) & (t < (starts + lengths)[:, None])  # (B, W)
            x2d = x2d.masked_fill(frame_mask[:, :, None, None], 0.0)
            x3d = x3d.masked_fill(frame_mask[:, :, None, None], 0.0)

        data["x3d"] = x3d
        data["x2d"] = x2d

    else:
        # assert "loss_mask" in data
        x2d = data["x2d"].to(device).float()

        data["target_pose"] = (x2d - x2d[:, :, 0:1]) * offsets_sum
        data["root"]        = x2d[:, :, 0] * offsets_sum

        if augment_dict["single_ablation"]:
            mask = torch.rand(x2d.shape[0], device=device) < augment_dict["single_ablation"]
            n_masked = int(mask.sum())
            if n_masked > 0:
                masked_idx = mask.nonzero(as_tuple=True)[0]
                keypt_to_ablate = torch.randint(
                    0, x2d.shape[-2], (n_masked,), device=device,
                )
                x2d[masked_idx, :, keypt_to_ablate, :] = 0


        if augment_dict["kpt_shuffle"]:
            B, W, N_K, D = x2d.shape
            rand_ind = torch.rand(B, N_K, device=device).argsort(dim=-1)
            x2d = torch.gather(
                x2d, dim=2,
                index=rand_ind[:, None, :, None].expand(-1, W, -1, D),
            )

        # [yw] temporal masking
        # mode 1: mask first 25 frames, predict the next 26
        if augment_dict.get("temporal_mask_past"):
            mask = torch.rand(x2d.shape[0], device=device) < augment_dict["temporal_mask_past"]
            if mask.any():
                x2d[mask, :25] = 0

        # mode 2: mask first 26 frames, predict the last 25
        if augment_dict.get("temporal_mask_future"):
            mask = torch.rand(x2d.shape[0], device=device) < augment_dict["temporal_mask_future"]
            if mask.any():
                x2d[mask, :26] = 0

        # mode 3: mask up to max_len consecutive frames in the interior
        if augment_dict.get("temporal_mask_random"):
            prob, max_len = augment_dict["temporal_mask_random"]
            max_len = int(max_len)
            B, W = x2d.shape[:2]
            apply = torch.rand(B, device=device) < prob                          # (B,)
            lengths = torch.randint(1, max_len + 1, (B,), device=device)
            starts = torch.randint(1, W - max_len, (B,), device=device)
            t = torch.arange(W, device=device)[None, :]                          # (1, W)
            frame_mask = apply[:, None] & (t >= starts[:, None]) & (t < (starts + lengths)[:, None])  # (B, W)
            x2d = x2d.masked_fill(frame_mask[:, :, None, None], 0.0)

        data["x2d"] = x2d

    return data

# def prepare_batch_2d_bespoke(
#     data: Dict,
#     augment_dict: Optional[Dict],
#     offsets_sum: float,
#     device: str = "cuda",
# ):
 
#     if "x2d" not in data and "pose" in data:
#         # ---- path A: raw pose -> midfwd-aligned x3d -----------------
#         pose = data["pose"]
#         if torch.is_tensor(pose):
#             pose = pose.detach().cpu().numpy()
#         pose = np.asarray(pose, dtype=np.float32)
 
#         # qrot_np needs 3D vectors; pad z=0 if input is plain 2D pose.
#         if pose.shape[-1] == 2:
#             pose = np.concatenate(
#                 [pose, np.zeros_like(pose[..., :1])], axis=-1
#             )
#         assert pose.shape[-1] == 3, (
#             f"path A expects pose of shape (B, W, K, 2 or 3), got {pose.shape}"
#         )
 
#         N, window, N_KEYPTS, _ = pose.shape
#         half_w = window // 2
 
#         root_xy = pose[:, half_w, 0, :2].copy()              # (N, 2)
#         pose_centered = pose.copy()
#         pose_centered[..., :2] -= root_xy[:, None, None, :]  # (N, W, K, 3)
 
#         forward = pose[:, half_w, 1, :2] - pose[:, half_w, 0, :2]    # (N, 2)
#         forward /= np.linalg.norm(forward, axis=-1, keepdims=True) + 1e-8
#         yaw = -np.arctan2(forward[:, 1], forward[:, 0])[..., None]   # (N, 1)
 
#         # Quaternion for rotation about +z by `yaw`. Quaternion layout is
#         # (w, x, y, z); we set w = cos(yaw/2) and z = sin(yaw/2).
#         fwd_q = np.zeros((N, 4), dtype=np.float32)
#         fwd_q[:, [-1, 0]] = np.concatenate(
#             [np.sin(yaw / 2), np.cos(yaw / 2)], axis=-1,
#         )
#         fwd_q_w = np.repeat(fwd_q[:, None, :], window, axis=1)
#         fwd_q_joints = fwd_q_w[:, :, None, :].repeat(N_KEYPTS, axis=2)
#         pose_aligned = qtn.qrot_np(
#             fwd_q_joints.reshape(-1, 4),
#             pose_centered.reshape(-1, 3),
#         ).reshape(N, window, N_KEYPTS, 3)
 
#         x3d = torch.from_numpy(pose_aligned).to(device) / offsets_sum
 
#         if augment_dict["single_ablation"]:
#             mask = torch.rand(x3d.shape[0]) < augment_dict["single_ablation"]
#             keypt_to_ablate = torch.randint(
#                 0, x3d.shape[-2], (mask.sum(),), device=device,
#             )
#             x3d[mask][..., keypt_to_ablate, :] = 0
 
#         if augment_dict["2d_td"]:
#             mask = torch.rand(x3d.shape[0]) < augment_dict["2d_td"]
#             x3d[mask, :, :, 2] = 0
 
#         if augment_dict["kpt_shuffle"]:
#             B, W, N_K, D = x3d.shape
#             rand_ind = torch.rand(B, N_K).to(device).argsort(dim=-1)
#             idx = rand_ind[:, None, :, None].expand(-1, W, -1, D)
#             x3d = torch.gather(x3d, dim=2, index=idx)
 
#         data["x3d"] = x3d
#         data["x2d"] = x3d[..., :2].clone()
#         x2d = data["x2d"].clone()
#         data["target_pose"] = (x2d.clone() - x2d.clone()[:, :, 0:1]) * offsets_sum
#         data["root"] = x2d[:, :, 0].clone() * offsets_sum
 
#     else:
#         x2d = data["x2d"].to(device).float()
 
#         if augment_dict["single_ablation"]:
#             mask = torch.rand(x2d.shape[0]) < augment_dict["single_ablation"]
#             keypt_to_ablate = torch.randint(
#                 0, x2d.shape[-2], (mask.sum(),), device=device,
#             )
#             x2d[mask][..., keypt_to_ablate, :] = 0
 
#         # 2d_td is a no-op on x2d directly: there is no z channel to zero.
 
#         if augment_dict["kpt_shuffle"]:
#             B, W, N_K, D = x2d.shape
#             rand_ind = torch.rand(B, N_K).to(device).argsort(dim=-1)
#             idx = rand_ind[:, None, :, None].expand(-1, W, -1, D)
#             x2d = torch.gather(x2d, dim=2, index=idx)
 
#         data["x2d"] = x2d.clone()
#         data["target_pose"] = (x2d.clone() - x2d.clone()[:, :, 0:1]) * offsets_sum
#         data["root"] = x2d[:, :, 0].clone() * offsets_sum
#         # data["x3d"] = torch.cat(
#             # [x2d, torch.zeros_like(x2d[..., :1])], dim=-1,
#         # )
 
#     return data


def get_x3d_from_data(x6d, offsets, root, offsets_sum, kinematic_tree):
    is_default_offsets = len(offsets.shape) == 2
    reshaped_x6d = x6d.reshape((-1,) + x6d.shape[-2:])
    offsets = offsets / offsets_sum
    if not is_default_offsets:
        offsets = offsets.reshape((-1,) + offsets.shape[-2:])
    root = root.reshape((-1, 3)) / offsets_sum
    x3d = fwd_kin_cont6d_torch(
        reshaped_x6d,
        kinematic_tree,
        offsets,
        root_pos=root,
        do_root_R=True,
        eps=1e-8,
    ).reshape(x6d.shape[:-1] + (3,))

    return x3d
