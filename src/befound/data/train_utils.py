from befound.data import fwd_kin_cont6d_torch
from typing import Dict, List
import torch


def prepare_batch(
    data: Dict,
    augment_dict: Dict,
    offsets_sum: float,
    kinematic_tree: List[int],
    device: str = "cuda",
    get_2d: bool = False,
):
    B, W, N_K, _ = data["x6d"].shape
    if "offsets" in data.keys():
        is_default_offsets = len(data["offsets"].shape) == 3

    for k, v in data.items():
        if k == "offsets" and is_default_offsets:
            data[k] = v[0].to(device)
        else:
            data[k] = v.to(device)

    offsets = data["offsets"].clone()
    if augment_dict.get("offset_noise"):
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

    if augment_dict.get("kpt_ablation"):
        max_ablate = int(augment_dict["kpt_ablation"]["max_ablate"])
        mask = torch.rand(B, device=device) < augment_dict["kpt_ablation"]["prob"]
        counts = torch.randint(1, max_ablate + 1, (B,), device=device)
        counts[~mask] = 0
        perm = torch.rand(B, N_K, device=device).argsort(dim=1)
        rank_mask = torch.arange(N_K, device=device).expand(B, N_K) < counts[:, None]

        keypt_mask = torch.zeros(B, N_K, dtype=torch.bool, device=device)
        keypt_mask.scatter_(1, perm, rank_mask)
        x3d = x3d.masked_fill(keypt_mask[:, None, :, None], 0)
        # keypt_mask = keypt_mask[:, None].expand(-1, W, -1)
        # x3d[mask][keypt_mask] = 0
        
        # keypt_to_ablate = torch.randint(0, x3d.shape[-2], (mask.sum(),), device=device)
        # x3d[mask][..., keypt_to_ablate, :] = 0

    if augment_dict.get("2d_td"):
        mask = torch.rand(x3d.shape[0], device=device) < augment_dict["2d_td"]
        x3d[mask, :, :, 2] = 0

    if augment_dict.get("kpt_shuffle"):
        rand_ind = torch.rand(B, N_K, device=device).argsort(dim=-1)
        idx = rand_ind[:, None, :, None].expand(-1, W, -1, x3d.shape[-1])
        x3d = torch.gather(x3d, dim=2, index=idx)


    temp_mask_keys = ["temporal_mask_past", "temporal_mask_future", "temporal_mask_interior"]
    for k in temp_mask_keys:
        if not bool(augment_dict.get(k)):
            if k == "temporal_mask_interior":
                augment_dict[k] = {"max_ablate": 0, "prob": 0.0}
            else:
                augment_dict[k] = 0.0

    try: 
        temporal_mask_prob = augment_dict["temporal_mask_past"] + augment_dict["temporal_mask_future"] + augment_dict["temporal_mask_interior"]["prob"]
        assert temporal_mask_prob <= 1.0
    except:
        raise ValueError("`temporal_mask_past` and `temporal_mask_future` and `temporal_mask_interior` cannot sum to more than 1.0")

    if temporal_mask_prob > 0:
        min_cutoff = 0
        max_cutoff = augment_dict["temporal_mask_past"]
        temp_mask_rand = torch.rand(x3d.shape[0],device=x3d.device) # (B,)
        # mode 1: mask first half of frames, predict the last half
        if augment_dict.get("temporal_mask_past")>0:
            mask = temp_mask_rand < max_cutoff
            if mask.any():
                x3d[mask, :W//2] = 0

            min_cutoff += augment_dict["temporal_mask_past"]

        # mode 2: mask second half of frames, predict the first half
        if augment_dict.get("temporal_mask_future")>0:
            max_cutoff += augment_dict["temporal_mask_future"]
            mask = (temp_mask_rand >= min_cutoff) & (temp_mask_rand < max_cutoff)
            if mask.any():
                x3d[mask, W//2:] = 0

            min_cutoff += augment_dict["temporal_mask_future"]

        # mode 3: mask up to max_len consecutive frames in the interior
        if (augment_dict.get("temporal_mask_interior")["max_ablate"]>0) and (augment_dict["temporal_mask_interior"]["prob"]>0):
            max_cutoff += augment_dict["temporal_mask_interior"]["prob"]
            mask = (temp_mask_rand >= min_cutoff) & (temp_mask_rand < max_cutoff) # (B,)

            max_ablate = int(augment_dict["temporal_mask_interior"]["max_ablate"])
            lengths = torch.randint(1, max_ablate + 1, (B,), device=x3d.device)
            starts = torch.randint(1, W - max_ablate, (B,), device=x3d.device)
            t = torch.arange(W, device=x3d.device)[None, :] # (1, W)
            frame_mask = mask[:, None] & (t >= starts[:, None]) & (t < (starts + lengths)[:, None]) # (B, W)
            x3d = x3d.masked_fill(frame_mask[:, :, None, None], 0.0)

    if get_2d:
        data["x2d"] = x3d[...,:2].clone()
    else:
        data["x3d"] = x3d.clone()
    return data


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
