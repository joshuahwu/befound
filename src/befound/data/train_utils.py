from befound.data import (
    fwd_kin_cont6d_torch,
    inv_kin_torch,
    get_frame_yaw_torch,
    get_angle2D_torch,
    get_segment_len_torch,
)
from befound.data.constants import DATA_FPS
from typing import Dict, List, Union, Optional
import torch
import befound.data.quaternion as qtn


def _augment_enabled(augment_dict: Dict, key: str) -> bool:
    """True if augment_dict[key] is a "real" (active) config -- not missing,
    not None, and not a dict whose values are all None. The last case covers
    the common YAML way of fully disabling a nested augmentation block, e.g.
    `fps: {min_fps: null, max_fps: null}` as an alternative to `fps: null`;
    without this, a non-empty dict of Nones is truthy and the augmentation
    branch runs, then crashes doing arithmetic on the None values inside it.
    """
    value = augment_dict.get(key)
    if value is None:
        return False
    if isinstance(value, dict) and any(v is None for v in value.values()):
        return False
    return True


def _sample_view_rotation(
    batch_size: int, device: str, pole_concentration: float = 4.0
) -> torch.Tensor:
    """Random 3D rotation (unit quaternion, shape (batch_size, 4)) whose
    viewing axis -- where the canonical +z axis ends up pointing -- is biased
    toward the poles (+z/-z), with a uniform full-360-degree azimuth for that
    axis and an independent, uniform spin about it. Meant to simulate
    realistic rodent-recording camera placements (top-down/bottom-up are the
    common ones; side-on views are rare), unlike a fully isotropic SO(3)
    sample which would spend most of its mass on oblique angles.

    pole_concentration > 1 biases cos(theta) (theta = polar angle from +z)
    toward +-1; pole_concentration == 1 reduces to a uniform-on-sphere axis
    (the original, fully isotropic sampler).
    """
    u = torch.rand(batch_size, device=device)
    sign = torch.where(
        torch.rand(batch_size, device=device) < 0.5,
        torch.full((batch_size,), -1.0, device=device),
        torch.full((batch_size,), 1.0, device=device),
    )
    cos_theta = sign * u.pow(1.0 / pole_concentration)
    sin_theta = torch.sqrt((1 - cos_theta**2).clamp(min=0))
    phi = torch.rand(batch_size, device=device) * 2 * torch.pi

    axis = torch.stack(
        [sin_theta * torch.cos(phi), sin_theta * torch.sin(phi), cos_theta], dim=-1
    )
    z_hat = torch.tensor([0.0, 0.0, 1.0], device=device).expand(batch_size, 3)

    # qbetween(v0, v1) is singular when v1 is antipodal to v0 (here, axis ~=
    # -z_hat) -- since this sampler concentrates mass near -z as well as +z,
    # that case comes up often enough to need an explicit fallback: a
    # 180-degree rotation about any axis perpendicular to z_hat (the x-axis,
    # here) also takes z_hat to -z_hat.
    dot = (z_hat * axis).sum(dim=-1)
    antipodal = dot < (-1.0 + 1e-6)
    q_axis = qtn.qbetween(z_hat, axis)
    fallback = torch.tensor([0.0, 1.0, 0.0, 0.0], device=device).expand(batch_size, 4)
    q_axis = torch.where(antipodal[:, None], fallback, q_axis)

    psi = torch.rand(batch_size, device=device) * 2 * torch.pi
    q_spin = torch.stack(
        [
            torch.cos(psi / 2),
            torch.zeros_like(psi),
            torch.zeros_like(psi),
            torch.sin(psi / 2),
        ],
        dim=-1,
    )

    return qtn.qnormalize(qtn.qmul(q_axis, q_spin))


def _center_window_query_positions(
    W: int, native_fps: float, target_fps: torch.Tensor, model_window: int
) -> torch.Tensor:
    """Fractional native-frame positions to sample for interpolate_center_window.
    See its docstring for what these positions mean."""
    device, dtype = target_fps.device, target_fps.dtype
    center = (W - 1) / 2.0
    k = torch.arange(model_window, device=device, dtype=dtype) - (model_window // 2)
    step = native_fps / target_fps  # (B,) native frames per target-fps step
    positions = center + k[None, :] * step[:, None]  # (B, model_window)
    return positions.clamp(0, W - 1)


def interpolate_center_window(
    pose: torch.Tensor,
    native_fps: float,
    target_fps: torch.Tensor,
    model_window: int,
    mode: str = "linear",
) -> torch.Tensor:
    """
    Resample a (B, W, N, D) pose loaded at native_fps to a per-sample target
    fps, taking exactly model_window frames centered on the window's middle
    frame (i.e. the point at native index (W-1)/2).

    Each output frame k (0-indexed, centered so k=model_window//2 lands
    exactly on the window's center) is interpolated from the native sequence
    at fractional native-frame position:
        center + (k - model_window//2) * (native_fps / target_fps[b])
    so target_fps == native_fps reduces to an exact center crop (no blur, no
    smoothing, regardless of mode), and target_fps > native_fps ("upsample")
    pulls the model_window frames from a narrower slice of native time, per
    the fps augmentation.

    Parameters
    ----------
    pose : torch.Tensor
        (B, W, N, D) pose at native_fps, W frames covering the window's full
        native-fps span (as loaded from disk -- W = window_sizes[dataset]).
    native_fps : float
        This dataset's native frame rate.
    target_fps : torch.Tensor
        (B,) per-sample target frame rate to resample to.
    model_window : int
        Number of output frames (the model's fixed window size, e.g. 51).
    mode : str
        "linear" (default) or "cubic". "cubic" uses Catmull-Rom, a *local*
        cubic spline: each query point is interpolated from its 4 nearest
        native frames (P0, P1, P2, P3, where the query falls between P1 and
        P2) via a fixed cubic blending matrix. It passes exactly through the
        original samples and is C1-continuous, without requiring the O(W)
        tridiagonal solve a global natural cubic spline (e.g.
        scipy.interpolate.CubicSpline) would need per sequence -- that solve
        doesn't vectorize across batch/keypoints the way a gather does, so
        it's not a practical fit here. Frames beyond the window edges are
        clamped (edge-replicated) rather than extrapolated. Cost vs. linear:
        ~2x the gather traffic (4 taps instead of 2) and a handful of extra
        elementwise multiply-adds for the t^2/t^3 terms -- still negligible
        in absolute terms next to the model forward/backward pass and the
        quaternion IK/FK math already done on this same window in
        get_midfwd_rotation.

    Returns
    -------
    torch.Tensor
        (B, model_window, N, D), same dtype/device as `pose`.
    """
    B, W, N, D = pose.shape
    dtype = pose.dtype

    positions = _center_window_query_positions(W, native_fps, target_fps, model_window)

    def _gather(idx):
        idx_exp = idx[:, :, None, None].expand(-1, -1, N, D)
        return torch.gather(pose, 1, idx_exp)

    if mode == "linear":
        floor_idx = positions.floor().long()
        ceil_idx = (floor_idx + 1).clamp(max=W - 1)
        frac = (
            (positions - floor_idx.to(dtype)).unsqueeze(-1).unsqueeze(-1)
        )  # (B, model_window, 1, 1)

        pose_floor = _gather(floor_idx)
        pose_ceil = _gather(ceil_idx)

        return pose_floor * (1 - frac) + pose_ceil * frac

    elif mode == "cubic":
        i1 = positions.floor().long()
        t = (
            (positions - i1.to(dtype)).unsqueeze(-1).unsqueeze(-1)
        )  # (B, model_window, 1, 1)

        i0 = (i1 - 1).clamp(0, W - 1)
        i2 = (i1 + 1).clamp(0, W - 1)
        i3 = (i1 + 2).clamp(0, W - 1)

        P0, P1, P2, P3 = _gather(i0), _gather(i1), _gather(i2), _gather(i3)

        t2 = t * t
        t3 = t2 * t
        return 0.5 * (
            2 * P1
            + (-P0 + P2) * t
            + (2 * P0 - 5 * P1 + 4 * P2 - P3) * t2
            + (-P0 + 3 * P1 - 3 * P2 + P3) * t3
        )

    else:
        raise ValueError(
            f"Unknown interpolation mode: {mode!r} (expected 'linear' or 'cubic')"
        )


def prepare_batch(
    data: Dict,
    dataset,
    augment_dict: Dict,
    model_window: int = 51,
    device: str = "cuda",
    get_2d: bool = False,
):
    """
    Parameters
    ----------
    dataset : befound.data.data.RodentDiskDataset
        Supplies everything that used to be passed as separate args --
        dataset_names, kinematic_tree, body_scale, dataset_idx_to_n_keypts,
        skeleton_offsets, default_seg_len, window_size -- as attributes, so
        callers don't have to pull them off the dataset themselves (see
        befound.train.trainer.train_epoch/test_epoch).
    """
    dataset_names = dataset.dataset_names
    kinematic_tree = dataset.kinematic_tree
    dataset_idx_to_n_keypts = dataset.dataset_idx_to_n_keypts
    skeleton_offsets = dataset.skeleton_offsets
    default_offsets = dataset.get_default_offsets(device)
    window_sizes = dataset.window_size

    B, W_MAX, N_MAX, D = data["pose"].shape
    W = model_window

    data = {k: v.to(device) for k, v in data.items()}

    if torch.any(torch.isnan(data["pose"])).item():
        import pdb

        pdb.set_trace()

    data["x6d"] = torch.zeros((B, W, N_MAX, 6), dtype=torch.float32, device=device)
    data["root"] = torch.zeros((B, W, 3), dtype=torch.float32, device=device)
    if get_2d:
        data["x2d"] = torch.zeros((B, W, N_MAX, 2), dtype=torch.float32, device=device)
        data["target_pose"] = torch.zeros_like(data["x2d"])
    else:
        data["x3d"] = torch.zeros((B, W, N_MAX, 3), dtype=torch.float32, device=device)
        data["target_pose"] = torch.zeros_like(data["x3d"])

    if _augment_enabled(augment_dict, "rotation"):
        # Identity rotation's 6D continuous representation, used as the default
        # for every sample so data["R_view_6d"] is always well-defined even when
        # the "rotation" augmentation is disabled or skips a given sample.
        identity_6d = torch.tensor(
            [1.0, 0.0, 0.0, 0.0, 1.0, 0.0], dtype=torch.float32, device=device
        )
        data["rotation"] = identity_6d.expand(B, 6).clone()

    if _augment_enabled(augment_dict, "2d_project"):
        data["2d_project"] = torch.zeros((B, 1), dtype=torch.long, device=device)

    if _augment_enabled(augment_dict, "fps"):
        data["fps"] = torch.zeros((B, 1), dtype=torch.float32, device=device)

    has_dataset_id = "dataset_id" in data
    if has_dataset_id:
        unique_dataset_ids = torch.unique(data["dataset_id"])
    else:
        # Single-dataset loaders may omit dataset_id entirely; every sample
        # belongs to the one dataset in dataset_names.
        assert (
            len(dataset_names) == 1
        ), "dataset_id missing from batch but multiple dataset_names were given"
        unique_dataset_ids = torch.zeros(1, dtype=torch.long)

    for d_id in unique_dataset_ids:
        d_id_int = d_id.item()
        if has_dataset_id:
            mask = data["dataset_id"] == d_id_int
        else:
            mask = torch.ones(B, dtype=torch.bool, device=data["pose"].device)

        B_id = mask.sum()
        dataset_name = dataset_names[d_id_int]
        n_keypts_id = dataset_idx_to_n_keypts[d_id_int]
        pose_d_id = data["pose"][mask, :, :n_keypts_id]

        W_d_id = (
            window_sizes[dataset_name]
            if isinstance(window_sizes, dict)
            else window_sizes
        )
        pose_d_id = pose_d_id[:, :W_d_id]
        native_fps = DATA_FPS[dataset_name]

        if _augment_enabled(augment_dict, "fps"):
            min_fps = augment_dict["fps"]["min_fps"]
            max_fps = augment_dict["fps"]["max_fps"]
            target_fps = (
                torch.rand(B_id, device=pose_d_id.device) * (max_fps - min_fps)
                + min_fps
            )
            data["fps"][mask] = target_fps[:, None]
            interp_mode = augment_dict["fps"].get("mode", "linear")
        else:
            # No fps augmentation: resample at native_fps, which reduces to
            # an exact center crop to model_window frames (no interpolation,
            # so the choice of mode doesn't matter here).
            target_fps = torch.full((B_id,), float(native_fps), device=pose_d_id.device)
            interp_mode = "linear"

        pose_d_id = interpolate_center_window(
            pose_d_id, native_fps, target_fps, W, mode=interp_mode
        )

        x6d_d_id, root_d_id, offsets_d_id = get_midfwd_rotation(
            pose=pose_d_id,
            kinematic_tree=kinematic_tree[dataset_name],
            skeleton_offsets=skeleton_offsets[dataset_name].to(device),
        )
        # root_d_id = torch.clamp(root_d_id, min=-body_scale[dataset_name], max=body_scale[dataset_name])

        # print(f"Root 5% and 95% quantile:{torch.quantile(root_d_id.reshape(-1, 3), q=torch.tensor([0.0, 1.0], device=root_d_id.device), dim=0)}")
        data["x6d"][mask, :, :n_keypts_id] = x6d_d_id.clone()
        data["root"][mask] = torch.clamp(root_d_id, min=-1.0, max=1.0)  # already normalized to body scale

        # Input Pose
        x3d_d_id = fwd_kin_cont6d_torch(
            continuous_6d=x6d_d_id.reshape(-1, n_keypts_id, 6),
            kinematic_tree=kinematic_tree[dataset_name],
            offset=offsets_d_id.reshape(-1, n_keypts_id, 3),
            root_pos=root_d_id.reshape(-1, 3),
            do_root_R=True,
            eps=1e-8,
        ).reshape(pose_d_id.shape)

        if _augment_enabled(augment_dict, "rotation"):
            # Viewing axis biased toward +-z (top-down/bottom-up), full
            # 360-degree azimuth and spin -- see _sample_view_rotation.
            rand_quat = _sample_view_rotation(B_id, device, pole_concentration=augment_dict["rotation"]["pole_concentration"])

            q_expand = rand_quat[:, None, None, :].expand(-1, W, n_keypts_id, -1)
            x3d_d_id = qtn.qrot(q_expand, x3d_d_id)

            data["rotation"][mask] = qtn.quaternion_to_cont6d(rand_quat).clone()

        x3d_d_id = apply_masking(x3d_d_id, augment_dict)
        if _augment_enabled(augment_dict, "2d_project"):
            mask_2d = (
                torch.rand(x3d_d_id.shape[0], device=device)
                < augment_dict["2d_project"]["prob"]
            )
            x3d_d_id[mask_2d, :, :, 2] = 0
            data["2d_project"][mask] = mask_2d.type(torch.long)[:, None]

        if get_2d:
            data["x2d"][mask, :, :n_keypts_id] = x3d_d_id[..., :2].clone()
        else:
            data["x3d"][mask, :, :n_keypts_id] = x3d_d_id.clone()

        # Target Pose
        if default_offsets is not None:
            target_offsets = default_offsets[dataset_name]
        else:
            target_offsets = offsets_d_id
        # target_offsets is already body-scale-normalized: default_seg_len
        # is pre-divided by body_scale in RodentDiskDataset.__init__, and
        # offsets_d_id (the fallback) is derived from the already-normalized
        # pose_d_id. data["root"] is likewise already normalized, coming
        # from the normalized pose via get_midfwd_rotation above.

        data["target_pose"][mask, :, :n_keypts_id] = (
            fwd_kin_cont6d_torch(
                continuous_6d=x6d_d_id.reshape(-1, n_keypts_id, 6),
                kinematic_tree=kinematic_tree[dataset_name],
                offset=target_offsets,
                root_pos=torch.zeros((B_id * W, 3), dtype=torch.float32, device=device),
                do_root_R=True,
                eps=1e-8,
            )
            .reshape(pose_d_id.shape)
            .clone()
        )

        # if data["target_pose"][mask, :, :n_keypts_id].max()>1.0 or data["target_pose"][mask, :, :n_keypts_id].min()<-1.0:
        #     from neuroposelib import vis, read
        #     connectivity = read.connectivity_config(dataset.data_path + dataset_name + "/skeleton.yaml")

        #     pose_flagged = x3d_d_id[torch.where(x3d_d_id == x3d_d_id.min())[0]].squeeze()
        #     vis.pose.arena3D(pose_flagged.cpu().detach().numpy(),
        #                         connectivity=connectivity,
        #                         frames=[0],
        #                         N_FRAMES=pose_flagged.shape[0],
        #                         centered=False,
        #                         fps=target_fps[torch.where(x3d_d_id == x3d_d_id.min())[0]].item(),
        #                         SAVE_ROOT="/mnt/home/jwu10/working/befound/scripts/prep_flagged/",
        #                         VID_NAME="flagged_pose.mp4")
        #     import pdb; pdb.set_trace()

        #     assert root_d_id.max() <=1.0 and root_d_id.min() >= -1.0, "Root is not normalized to body scale"

    return data

def get_midfwd_rotation(pose, kinematic_tree, skeleton_offsets):
    device = pose.device
    B, W, N_K, D = pose.shape
    yaw = get_frame_yaw_torch(pose[:, W // 2], 0, 1)[..., None]

    root = pose[..., 0, :].clone()
    root -= root[:, W // 2, :].clone()[:, None, :]

    local_qtn = inv_kin_torch(
        pose.reshape((-1, N_K, D)),
        kinematic_tree,
        offset=skeleton_offsets,
        forward_indices=[1, 0],
    ).reshape(B, W, N_K, 4)

    fwd_qtn = torch.zeros((B, 4), dtype=yaw.dtype, device=device)
    fwd_qtn[:, [-1, 0]] = get_angle2D_torch(yaw / 2)
    fwd_qtn = fwd_qtn.unsqueeze(1).repeat(1, W, 1)
    local_qtn[..., 0, :] = qtn.qmul(fwd_qtn, local_qtn[..., 0, :])

    root = qtn.qrot(fwd_qtn, root)
    x6d = qtn.quaternion_to_cont6d(local_qtn)

    offsets = get_segment_len_torch(
        pose.reshape(-1, N_K, D),
        kinematic_tree,
        skeleton_offsets,
    ).reshape(B, W, N_K, D)

    return x6d, root, offsets


def apply_masking(x3d: torch.Tensor, augment_dict: Dict) -> torch.Tensor:
    """
    Apply kpt_ablation, kpt_shuffle, noise, and
    temporal_mask_{past,future,interior} augmentations to a single
    dataset's 3D pose tensor.

    Parameters
    ----------
    x3d : torch.Tensor
        Shape (B, W, N_K, 3)
    augment_dict : Dict
        Augmentation config with keys kpt_ablation, kpt_shuffle, noise,
        temporal_mask_past, temporal_mask_future, temporal_mask_interior
    """
    device = x3d.device
    B, W, N_K, _ = x3d.shape

    if _augment_enabled(augment_dict, "kpt_ablation"):
        max_ablate = int(augment_dict["kpt_ablation"]["max_ablate"])
        mask = torch.rand(B, device=device) < augment_dict["kpt_ablation"]["prob"]
        counts = torch.randint(1, max_ablate + 1, (B,), device=device)
        counts[~mask] = 0
        perm = torch.rand(B, N_K, device=device).argsort(dim=1)
        rank_mask = torch.arange(N_K, device=device).expand(B, N_K) < counts[:, None]

        keypt_mask = torch.zeros(B, N_K, dtype=torch.bool, device=device)
        keypt_mask.scatter_(1, perm, rank_mask)
        x3d = x3d.masked_fill(keypt_mask[:, None, :, None], 0)

    if augment_dict.get("kpt_shuffle"):
        rand_ind = torch.rand(B, N_K, device=device).argsort(dim=-1)
        idx = rand_ind[:, None, :, None].expand(-1, W, -1, x3d.shape[-1])
        x3d = torch.gather(x3d, dim=2, index=idx)

    if _augment_enabled(augment_dict, "noise"):
        # x3d is already body-scale-normalized (see RodentDiskDataset
        # .__getitem__), so magnitude is directly a std in units of
        # fraction-of-body-length -- the same magnitude means the same
        # relative jitter across every dataset.
        x3d = x3d + augment_dict["noise"]["magnitude"] * torch.randn_like(x3d)

    temporal_mask_past = (
        augment_dict["temporal_mask_past"]["prob"]
        if _augment_enabled(augment_dict, "temporal_mask_past")
        else 0.0
    )
    temporal_mask_future = (
        augment_dict["temporal_mask_future"]["prob"]
        if _augment_enabled(augment_dict, "temporal_mask_future")
        else 0.0
    )
    temporal_mask_interior = (
        augment_dict["temporal_mask_interior"]
        if _augment_enabled(augment_dict, "temporal_mask_interior")
        else {"max_ablate": 0, "prob": 0.0}
    )

    temporal_mask_prob = (
        temporal_mask_past + temporal_mask_future + temporal_mask_interior["prob"]
    )
    if temporal_mask_prob > 1.0:
        raise ValueError(
            "`temporal_mask_past` and `temporal_mask_future` and `temporal_mask_interior` cannot sum to more than 1.0"
        )

    if temporal_mask_prob > 0:
        min_cutoff = 0
        max_cutoff = temporal_mask_past
        temp_mask_rand = torch.rand(B, device=device)  # (B,)

        # mode 1: mask first half of frames, predict the last half
        if temporal_mask_past > 0:
            mask = temp_mask_rand < max_cutoff
            if mask.any():
                x3d[mask, : W // 2] = 0

            min_cutoff += temporal_mask_past

        # mode 2: mask second half of frames, predict the first half
        if temporal_mask_future > 0:
            max_cutoff += temporal_mask_future
            mask = (temp_mask_rand >= min_cutoff) & (temp_mask_rand < max_cutoff)
            if mask.any():
                x3d[mask, W // 2 :] = 0

            min_cutoff += temporal_mask_future

        # mode 3: mask up to max_ablate consecutive frames in the interior
        if (
            temporal_mask_interior["max_ablate"] > 0
            and temporal_mask_interior["prob"] > 0
        ):
            max_cutoff += temporal_mask_interior["prob"]
            mask = (temp_mask_rand >= min_cutoff) & (
                temp_mask_rand < max_cutoff
            )  # (B,)

            max_ablate = int(temporal_mask_interior["max_ablate"])
            lengths = torch.randint(1, max_ablate + 1, (B,), device=device)
            starts = torch.randint(1, W - max_ablate, (B,), device=device)
            t = torch.arange(W, device=device)[None, :]  # (1, W)
            frame_mask = (
                mask[:, None]
                & (t >= starts[:, None])
                & (t < (starts + lengths)[:, None])
            )  # (B, W)
            x3d = x3d.masked_fill(frame_mask[:, :, None, None], 0.0)

    return x3d


# def get_x3d_from_data(x6d, offsets, root, body_scale, kinematic_tree):
#     if not body_scale:
#         body_scale = 1
#     is_default_offsets = len(offsets.shape) == 2
#     reshaped_x6d = x6d.reshape((-1,) + x6d.shape[-2:])
#     offsets = offsets / body_scale
#     if not is_default_offsets:
#         offsets = offsets.reshape((-1,) + offsets.shape[-2:])
#     root = root.reshape((-1, 3)) / body_scale
#     x3d = fwd_kin_cont6d_torch(
#         reshaped_x6d,
#         kinematic_tree,
#         offsets,
#         root_pos=root,
#         do_root_R=True,
#         eps=1e-8,
#     ).reshape(x6d.shape[:-1] + (3,))

#     return x3d
