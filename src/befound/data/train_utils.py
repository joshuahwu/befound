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
