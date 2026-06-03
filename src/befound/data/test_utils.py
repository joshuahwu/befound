
from __future__ import annotations

import os
from pathlib import Path
from typing import List

import numpy as np
import torch
from tqdm import trange
from numpy.lib.stride_tricks import sliding_window_view

from befound.get.get import get_model
from befound.params import read as params_read
from befound.data.train_utils import prepare_batch, prepare_batch_2d_bespoke
from befound.data.constants import OFFSETS_3D_MABE22_SUM

from sklearn.decomposition import PCA

from collections import defaultdict

@torch.no_grad()
def encode_decode_all(
    model,
    loader,
    augment_dict,
    offsets_sum,
    kinematic_tree,
    get_2d,
    device="cuda",
    keys=None,
    to_cpu=True,
):
    """Run model(prepare_batch(...)) over the whole loader; concat each
    output key along dim 0. Returns dict[str, np.ndarray | torch.Tensor].

    Skips non-tensor outputs (e.g. `beta_dist` when prior='beta').
    """
    model.eval()
    bins = defaultdict(list)
    for data in loader:
        data = {k: v.to(device, non_blocking=True) for k, v in data.items()}
        data = prepare_batch(
            data=data, augment_dict=augment_dict, offsets_sum=offsets_sum,
            kinematic_tree=kinematic_tree, device=device, get_2d=get_2d,
        )
        data_o = model(data)
        for k, v in data_o.items():
            if not torch.is_tensor(v):           # skip beta_dist etc.
                continue
            if keys is not None and k not in keys:
                continue
            bins[k].append(v.detach().cpu() if to_cpu else v.detach())

    cat = {k: torch.cat(vs, dim=0) for k, vs in bins.items()}
    return {k: v.numpy() for k, v in cat.items()} if to_cpu else cat


# @torch.no_grad()
# def encode_all_trajectories(
#     model,
#     keypoints: np.ndarray,   # (N, T, K, 2) float32, already reindexed
#     offsets_sum: float,
#     window: int = 51,
#     batch_size: int = 512,
#     device: str = "cuda",
#     pad_mode: str = "edge",
# ) -> np.ndarray:
#     """
#     Return (N, T, d_latent) float32 of per-frame latent means.

#     One mouse trajectory is processed at a time to cap peak host RAM. Within
#     each trajectory `batch_size` windows are pushed through the model at a
#     time. We pass numpy windows straight into prepare_batch_2d_bespoke (path
#     A); it handles numpy->torch conversion and the midfwd alignment itself.
#     """
#     def sliding_windows(seq: np.ndarray, window: int = WINDOW, pad_mode: str = "edge"):
#         """
#         seq: (T, K, D) numpy array for one mouse trajectory.

#         Returns: (T, window, K, D) contiguous array where windows[t] is centered
#         on frame t. Edge padding replicates the first/last frame (better than
#         zero-pad here, because 0 is already the `missing keypoint` marker in
#         MABe and would collide with true missing data).
#         """
#         assert seq.ndim == 3, f"expected (T, K, D), got {seq.shape}"
#         half = window // 2
#         padded = np.pad(seq, ((half, half), (0, 0), (0, 0)), mode=pad_mode)
#         # view shape after sliding on axis 0: (T, K, D, window)
#         view = sliding_window_view(padded, window_shape=window, axis=0)
#         # move window axis to position 1: (T, window, K, D)
#         return np.ascontiguousarray(np.moveaxis(view, -1, 1))

#     N, T = keypoints.shape[:2]
#     model.eval()

#     z_all = None  # allocated lazily once we see d_latent

#     for n in trange(N, desc="encode trajectories"):
#         win = sliding_windows(keypoints[n], window=window, pad_mode=pad_mode)
#         # win: (T, W, K, 2) float32

#         seq_z = []
#         for start in range(0, T, batch_size):
#             end = min(start + batch_size, T)
            
#             augment_dict = {
#                     "2d_td": None,
#                     "kpt_shuffle": None,
#                     "offset_noise": None,
#                     "single_ablation": None,
#                 }

#             data = {"pose": win[start:end]}                     # numpy, path A
#             data = prepare_batch_2d_bespoke(
#                 data=data,
#                 augment_dict=augment_dict,
#                 offsets_sum=offsets_sum,
#                 device=device,
#             )
#             data_o = model(data)
#             for k, v in data_o.items():
#                 if not torch.is_tensor(v):           # skip beta_dist etc.
#                     continue
#                 if keys is not None and k not in keys:
#                     continue
#                 bins[k].append(v.detach().cpu() if to_cpu else v.detach())

#         cat = {k: torch.cat(vs, dim=0) for k, vs in bins.items()}
#         return {k: v.numpy() for k, v in cat.items()} if to_cpu else cat


# def sliding_windows(seq: np.ndarray, window: int = WINDOW,
#                     stride: int = 1, pad_mode: str = "edge"):
#     """
#     seq: (T, K, D) numpy array for one mouse trajectory.

#     Returns: (M, window, K, D) contiguous array where windows[i] is centred
#     on frame `i * stride`. M = len(range(0, T, stride)) = ceil(T / stride).
#     Edge padding replicates the first/last frame (better than zero-pad
#     here, because 0 is already the `missing keypoint` marker in MABe and
#     would collide with true missing data).
#     """
#     assert seq.ndim == 3, f"expected (T, K, D), got {seq.shape}"
#     half = window // 2
#     padded = np.pad(seq, ((half, half), (0, 0), (0, 0)), mode=pad_mode)
#     view = sliding_window_view(padded, window_shape=window, axis=0)  # (T,K,D,W)
#     view = np.moveaxis(view, -1, 1)                                  # (T,W,K,D)
#     if stride > 1:
#         view = view[::stride]                                        # (M,W,K,D) still a view
#     return np.ascontiguousarray(view)

def sliding_windows(seq: np.ndarray, window: int = 51,
                    stride: int = 1, pad_mode: str = "edge"):
    """
    seq: (T, K, D) numpy array for one mouse trajectory.

    Returns: (M, window, K, D) contiguous array where windows[i] is centred
    on frame `i * stride`. M = len(range(0, T, stride)) = ceil(T / stride).
    Edge padding replicates the first/last frame (better than zero-pad
    here, because 0 is already the `missing keypoint` marker in MABe and
    would collide with true missing data).
    """
    assert seq.ndim == 3, f"expected (T, K, D), got {seq.shape}"
    half = window // 2
    padded = np.pad(seq, ((half, half), (0, 0), (0, 0)), mode=pad_mode)
    view = sliding_window_view(padded, window_shape=window, axis=0)  # (T,K,D,W)
    view = np.moveaxis(view, -1, 1)                                  # (T,W,K,D)
    if stride > 1:
        view = view[::stride]                                        # (M,W,K,D) still a view
    return np.ascontiguousarray(view)


@torch.no_grad()
def encode_decode_all_trajectories(
    model,
    keypoints: np.ndarray,         # (N, T, K, 2) float32, already reindexed
    offsets_sum: float,
    window: int = 51,
    batch_size: int = 512,
    stride: int = 1,
    device: str = "cuda",
    pad_mode: str = "edge",
    keys=None,                     # None = collect every tensor-valued key
) -> dict:
    """
    Run encode + decode on a strided set of windows over every trajectory;
    return a dict mapping each model output key to an (N, M, ...) float32
    ndarray.

    M = ceil(T / stride) -- one window per centre frame in
    `range(0, T, stride)`. Decoder outputs keep their full window axis,
    e.g. `x2d` lands at (N, M, window, K, 2). Use `stride=1` for dense
    per-frame coverage; `stride=window` for non-overlapping tiles.

    Only tensor-valued outputs are collected; non-tensor returns (e.g.
    `beta_dist` from prior='beta' models) are silently skipped. Pass
    `keys={"mu"}` etc. to restrict collection and save host RAM.
    """
    N, T = keypoints.shape[:2]
    M = len(range(0, T, stride))
    model.eval()

    augment_dict = {
        "2d_td": None, "kpt_shuffle": None,
        "offset_noise": None, "single_ablation": None,
    }

    out: dict = {}  # key -> (N, M, ...) ndarray, lazy-allocated

    for n in trange(N, desc="encode+decode trajectories"):
        win = sliding_windows(keypoints[n], window=window,
                              stride=stride, pad_mode=pad_mode)
        # win: (M, W, K, 2)

        for start in range(0, M, batch_size):
            end = min(start + batch_size, M)
            data = {"pose": win[start:end]}
            data = prepare_batch_2d_bespoke(
                data=data, augment_dict=augment_dict,
                offsets_sum=offsets_sum, device=device,
            )

            if "target_pose" not in out:
                out["target_pose"] = np.empty((N, M, *data["target_pose"].shape[1:]), dtype=np.float32)
            
            out["target_pose"][n, start:end] = data["target_pose"].cpu().numpy()  # (M, W, K, 2)

            data_o = model(data)
            for k, v in data_o.items():
                if not torch.is_tensor(v):                # skip beta_dist
                    continue
                if keys is not None and k not in keys:
                    continue
                arr = v.detach().float().cpu().numpy()    # (B, ...)
                if k not in out:
                    out[k] = np.empty((N, M, *arr.shape[1:]), dtype=np.float32)
                out[k][n, start:end] = arr

    return out