from neuroposelib import read
import numpy as np
import befound.data.quaternion as qtn
from typing import Optional, Type, Union, List, Dict
from torch.utils.data import Dataset, Sampler
import torch
from numpy.lib.stride_tricks import sliding_window_view
from tqdm import trange
import numpy as np
from typing import List
import torch
import h5py
from befound.data.constants import OFFSETS_3D, BODY_SCALE, get_window_size, SEQUENCE_SCALE #OFFSETS_3D_SUM


def inv_kin_torch(
    pose: torch.Tensor,
    kinematic_tree: Union[List, torch.Tensor],
    offset: torch.Tensor,
    forward_indices: Union[List, torch.Tensor] = [0, 1],
):
    """
    Adapted from T2M-GPT (https://mael-zys.github.io/T2M-GPT/)
    [1] Zhang, Jianrong, et al. "Generating Human Motion From Textual
    Descriptions With Discrete Representations." Proceedings of the
    IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2023.
    """

    # Find forward root direction
    forward = pose[:, forward_indices[1], :] - pose[:, forward_indices[0], :]
    forward = forward / torch.norm(forward, dim=-1, keepdim=True)

    # Root Rotation
    target = torch.tensor([[1, 0, 0]], dtype=pose.dtype, device=pose.device).repeat(
        len(forward), 1
    )
    root_quat = qtn.qbetween(forward, target)

    local_quat = torch.zeros(
        pose.shape[:-1] + (4,), dtype=pose.dtype, device=pose.device
    )
    # root_quat[0] = torch.tensor(
    #     [[1.0, 0.0, 0.0, 0.0]], dtype=pose.dtype, device=pose.device
    # )
    root_quat[:, 0] = torch.nan_to_num(root_quat[:, 0], 1)
    root_quat[:, 1:] = torch.nan_to_num(root_quat[:, 1:], 0)
    local_quat[:, 0] = root_quat

    for chain in kinematic_tree:
        R = root_quat
        for i in range(len(chain) - 1):
            u = offset[chain[i + 1]].unsqueeze(0).repeat(len(pose), 1)
            v = pose[:, chain[i + 1]] - pose[:, chain[i]]
            v = v / torch.norm(v, dim=-1, keepdim=True)
            rot_u_v = qtn.qbetween(u, v)
            rot_u_v[:, 0] = torch.nan_to_num(rot_u_v[:, 0], 1)
            rot_u_v[:, 1:] = torch.nan_to_num(rot_u_v[:, 1:], 0)
            R_loc = qtn.qmul(qtn.qinv(R), rot_u_v)
            local_quat[:, chain[i + 1], :] = R_loc
            R = qtn.qmul(R, R_loc)

    return local_quat


def inv_kin(
    pose: np.ndarray,
    kinematic_tree: Union[List, np.ndarray],
    offset: np.ndarray,
    forward_indices: Union[List, np.ndarray] = [0, 1],
):
    """
    Adapted from T2M-GPT (https://mael-zys.github.io/T2M-GPT/)
    [1] Zhang, Jianrong, et al. "Generating Human Motion From Textual
    Descriptions With Discrete Representations." Proceedings of the
    IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2023.
    """

    # Find forward root direction
    forward = pose[:, forward_indices[1], :] - pose[:, forward_indices[0], :]
    forward = forward / np.linalg.norm(forward, axis=-1)[..., None]

    # Root Rotation
    target = np.array([[1, 0, 0]]).repeat(len(forward), axis=0)
    root_quat = qtn.qbetween_np(forward, target)

    local_quat = np.zeros(pose.shape[:-1] + (4,))
    # import pdb; pdb.set_trace()
    # root_quat[0] = np.array([[1.0, 0.0, 0.0, 0.0]])
    local_quat[:, 0] = root_quat
    for chain in kinematic_tree:
        R = root_quat
        for i in range(len(chain) - 1):
            u = offset[chain[i + 1]][None, ...].repeat(len(pose), axis=0)
            v = pose[:, chain[i + 1]] - pose[:, chain[i]]
            v = v / np.linalg.norm(v, axis=-1)[..., None]
            rot_u_v = qtn.qbetween_np(u, v)
            R_loc = qtn.qmul_np(qtn.qinv_np(R), rot_u_v)
            local_quat[:, chain[i + 1], :] = R_loc
            R = qtn.qmul_np(R, R_loc)

    return local_quat


def fwd_kin_cont6d(
    continuous_6D: np.ndarray,
    kinematic_tree: Union[List, np.ndarray],
    offset: np.ndarray,
    root_pos: np.ndarray,
    do_root_R: bool = True,
):
    # continuous_6D (batch_size, pose_num, 6)
    # pose (batch_size, pose_num, 3)
    # root_pos (batch_size, 3)

    pose = np.zeros(continuous_6D.shape[:-1] + (3,))
    pose[:, 0] = root_pos

    if len(offset.shape) == 2:
        offsets = np.moveaxis(np.tile(offset[..., None], continuous_6D.shape[0]), -1, 0)
    else:
        offsets = offset

    for chain in kinematic_tree:
        if do_root_R:
            matR = qtn.cont6d_to_matrix_np(continuous_6D[:, 0])
        else:
            matR = np.eye(3)[np.newaxis, :].repeat(len(continuous_6D), axis=0)
        for i in range(1, len(chain)):
            matR = np.matmul(matR, qtn.cont6d_to_matrix_np(continuous_6D[:, chain[i]]))
            offset_vec = offsets[:, chain[i]][..., np.newaxis]
            # print(matR.shape, offset_vec.shape)
            pose[:, chain[i]] = (
                np.matmul(matR, offset_vec).squeeze(-1) + pose[:, chain[i - 1]]
            )
    return pose


def fwd_kin_cont6d_torch(
    continuous_6d, kinematic_tree, offset, root_pos, do_root_R=True, eps=0
):
    # continuous_6d (batch_size, joints_num, 6)
    # joints (batch_size, joints_num, 3)
    # root_pos (batch_size, 3)

    if len(offset.shape) == 2:
        offsets = offset.expand(continuous_6d.shape[0], -1, -1)
    else:
        offsets = offset

    pose = torch.zeros(continuous_6d.shape[:-1] + (3,), device=continuous_6d.device)
    pose[..., 0, :] = root_pos
    for chain in kinematic_tree:
        if do_root_R:
            matR = qtn.cont6d_to_matrix(continuous_6d[:, 0], eps=eps)
        else:
            matR = (
                torch.eye(3)
                .expand((len(continuous_6d), -1, -1))
                .detach()
                .to(continuous_6d.device)
            )
        for i in range(1, len(chain)):
            matR = torch.matmul(
                matR, qtn.cont6d_to_matrix(continuous_6d[:, chain[i]], eps=eps)
            )
            offset_vec = offsets[:, chain[i]].unsqueeze(-1)

            pose[:, chain[i]] = (
                torch.matmul(matR, offset_vec).squeeze(-1) + pose[:, chain[i - 1]]
            )
    return pose


def normalize_root(root, arena_size):
    """
    Normalize root Cartesian coordinates to be from (-1, 1) based on arena size
    """
    norm_root = root - arena_size[0]
    norm_root = 2 * norm_root / (arena_size[1] - arena_size[0]) - 1
    return norm_root


def inv_normalize_root(norm_root, arena_size):
    root = 0.5 * (norm_root + 1) * (arena_size[1] - arena_size[0])
    root += arena_size[0]
    return root


def get_speed_parts(pose, parts):
    """
    Get the (1) average root displacement,
    (2) average speed of the spine relative to the root,
    and (3) average speed of the limbs relative to the spine
    """
    print("Getting speed by body parts")
    root_spd = np.diff(pose[..., 0, :], n=1, axis=-2) ** 2  # (n_samples, window, 3)
    root_spd = np.sqrt(root_spd.sum(-1)).mean(-1)
    dxyz = np.zeros((len(root_spd), len(parts) + 1))
    dxyz[:, 0] = root_spd  # TODO: Put sum in sqrt

    centered_pose = pose - pose[..., 0:1, :]
    for i, part in enumerate(parts):
        if part[0] == 0:
            pose_part = centered_pose
        else:
            pose_part = centered_pose - centered_pose[:, part[0] : part[0] + 1, :]

        relative_dxyz = (
            np.diff(
                pose_part[..., part[1:], :],
                n=1,
                axis=-3,
            )
            ** 2
        ).sum(-1)
        dxyz[:, i + 1] = np.sqrt(relative_dxyz).mean(axis=(-1, -2))

    return dxyz


def get_speed_parts_torch(pose, parts):
    """
    Pytorch version
    Get the (1) average root displacement,
    (2) average speed of the spine relative to the root,
    and (3) average speed of the limbs relative to the spine
    """
    print("Getting speed by body parts")
    root_spd = (
        torch.diff(pose[..., 0, :], n=1, dim=-3, prepend=pose[..., 0:1, 0, :]) ** 2
    )
    dxyz = torch.zeros((len(root_spd), len(parts) + 1), device=pose.device)
    dxyz[:, 0] = torch.sqrt(root_spd.sum(dim=-1))

    centered_pose = pose - pose[:, 0:1, :]

    for i, part in enumerate(parts):
        pose_part = centered_pose - centered_pose[:, part[0] : part[0] + 1, :]
        relative_dxyz = (
            torch.diff(
                pose_part[:, part[1:], :],
                n=1,
                dim=0,
                prepend=pose_part[0:1, part[1:], :],
            )
            ** 2
        ).sum(dim=-1)
        dxyz[:, i + 1] = torch.sqrt(relative_dxyz).mean(dim=-1)

    return dxyz


def get_window_indices(ids, stride, window):
    """
    Get full indices of an array broken up by sliding windows
    """
    print("Calculating windowed indices ...")
    window_inds = []
    frame_idx = np.arange(len(ids), dtype=int)
    id_diff = np.diff(ids, prepend=ids[0])
    id_change = np.concatenate([[0], np.where(id_diff != 0)[0], [len(ids)]])
    for i in trange(0, len(id_change) - 1):
        if (id_change[i + 1] - id_change[i]) >= window:
            strided_data = sliding_window_view(
                frame_idx[id_change[i] : id_change[i + 1], ...],
                window_shape=window,
                axis=0,
            )[::stride, ...]

            window_inds += [torch.tensor(strided_data, dtype=int)]

            if strided_data.shape[0] > 1:
                assert (
                    np.moveaxis(strided_data[1, ...], -1, 0)
                    - frame_idx[
                        id_change[i] + stride : id_change[i] + window + stride, ...
                    ]
                ).sum() == 0
        else:
            print(
                "ID {} length smaller than window size - skipping ...".format(
                    ids[id_change[i]]
                )
            )

    window_inds = torch.cat(window_inds, dim=0)

    return window_inds


def get_frame_yaw(pose, root_i=0, front_i=1):
    """
    Get yaw of given segment in radians
    """
    forward = pose[:, front_i, :] - pose[:, root_i, :]
    forward = forward / np.linalg.norm(forward, axis=-1)[..., None]
    yaw = -np.arctan2(forward[:, 1], forward[:, 0])
    return yaw


def get_frame_yaw_torch(pose: torch.Tensor, root_i=0, front_i=1) -> torch.Tensor:
    """
    Get yaw of given segment in radians
    """
    forward = pose[:, front_i, :] - pose[:, root_i, :]
    forward = forward / torch.linalg.norm(forward, dim=-1, keepdim=True)
    yaw = -torch.atan2(forward[:, 1], forward[:, 0])
    return yaw


def get_heading2D(pose, root_i=0, front_i=1):
    """
    NOT USED

    Get yaw of given segment as [sin(angle), cos(angle)]
    i.e., coordinates on a unit circle
    """
    yaw = get_frame_yaw(pose, root_i, front_i)
    heading2D = get_angle2D(yaw[:, None])
    heading_change = np.diff(heading2D, n=1, axis=0, prepend=heading2D[0:1, :])

    return np.append(heading2D, heading_change, axis=-1)


def get_angle2D(angle):  # sin is first, then cos
    """
    Given angles in radians, return [sin(angle), cos(angle)]
    i.e., coordinates on a unit circle
    """
    angle2D = np.concatenate([np.sin(angle)[:, None], np.cos(angle)[:, None]], axis=-1)
    angle2D = angle2D.reshape(angle.shape[:-1] + (-1,))
    return angle2D


def get_angle2D_torch(angle: torch.Tensor) -> torch.Tensor:  # sin is first, then cos
    """
    Given angles in radians, return [sin(angle), cos(angle)]
    i.e., coordinates on a unit circle
    """
    angle2D = torch.cat([torch.sin(angle)[:, None], torch.cos(angle)[:, None]], dim=-1)
    angle2D = angle2D.reshape(angle.shape[:-1] + (-1,))
    return angle2D


def get_angle_from_2D(angle2D):
    """
    Given coordinates on a unit circle, return angle in radians
    """
    angle2D = angle2D.reshape(angle2D.shape[0], -1, 2)
    angles = np.arctan2(angle2D[..., 0], angle2D[..., 1])
    return angles


def get_segment_len(pose: np.ndarray, kinematic_tree: np.ndarray, offset: np.ndarray):
    """
    Get length of all segments in a pose defined by a kinematic tree
    """
    offset = offset.astype(pose.dtype)
    parents = [0] * len(offset)
    parents[0] = -1
    for chain in kinematic_tree:
        for j in range(1, len(chain)):
            parents[chain[j]] = chain[j - 1]

    offsets = np.moveaxis(np.tile(offset[..., None], pose.shape[0]), -1, 0)
    for i in range(1, offset.shape[0]):
        length = np.linalg.norm(pose[:, i, :] - pose[:, parents[i], :], axis=1)[
            ..., None
        ]
        length = np.clip(length, a_min=1e-3, a_max=None)
        offsets[:, i] = length * offsets[:, i]

    return offsets


def get_segment_len_torch(
    pose: torch.Tensor, kinematic_tree: torch.Tensor, offset: torch.Tensor
) -> torch.Tensor:
    """
    Get length of all segments in a pose defined by a kinematic tree
    """
    offset = offset.to(pose.dtype)
    parents = [0] * len(offset)
    parents[0] = -1
    for chain in kinematic_tree:
        for j in range(1, len(chain)):
            parents[chain[j]] = chain[j - 1]

    offsets = offset.unsqueeze(0).repeat(pose.shape[0], 1, 1)
    for i in range(1, offset.shape[0]):
        length = torch.linalg.norm(
            pose[:, i, :] - pose[:, parents[i], :], dim=1, keepdim=True
        )
        length = torch.clamp(length, min=1e-3)
        offsets[:, i] = length * offsets[:, i]

    return offsets


def get_speed_outliers(pose, threshold=2.25):
    """
    Find indices of frames in which the average speed is greater than the defined threshold
    """
    inds = np.arange(len(pose))

    avg_spd = np.diff(pose, n=1, axis=-3)
    avg_spd = np.sqrt((avg_spd**2).sum(axis=-1)).mean(axis=(-1, -2))
    outlier_frames = np.where(avg_spd > threshold)[0]
    print("Outlier frames above {}: {}".format(threshold, len(outlier_frames)))
    return outlier_frames

class BalancedDatasetSampler(Sampler[int]):
    """
    Samples window indices so each dataset is drawn ~equally in expectation,
    with replacement, without ever building a single len(dataset)-sized
    weights tensor (unlike WeightedRandomSampler, whose underlying
    torch.multinomial call hard-caps the number of categories at 2**24 --
    far below this class's typical combined window counts).

    Equivalent in distribution to a WeightedRandomSampler with per-window
    weight 1/counts[dataset_idx] (every window of a dataset sharing the same
    weight): picking a dataset uniformly at random, then a uniformly random
    window within it, is the same thing without the oversized weights
    tensor -- picking only among len(dataset_names) categories instead.
    """

    def __init__(
        self, entry_dataset_idx: List[int], num_datasets: int, num_samples: int
    ):
        dataset_idx = torch.as_tensor(entry_dataset_idx, dtype=torch.long)
        self.num_samples = num_samples
        self.dataset_indices = [
            torch.nonzero(dataset_idx == i, as_tuple=True)[0]
            for i in range(num_datasets)
        ]

    def __iter__(self):
        num_datasets = len(self.dataset_indices)
        chosen = torch.randint(0, num_datasets, (self.num_samples,))
        result = torch.empty(self.num_samples, dtype=torch.long)
        for d in range(num_datasets):
            mask = chosen == d
            n = int(mask.sum())
            if n == 0:
                continue
            pool = self.dataset_indices[d]
            local = torch.randint(0, len(pool), (n,))
            result[mask] = pool[local]
        yield from result.tolist()

    def __len__(self) -> int:
        return self.num_samples


class RodentDiskDataset(Dataset):
    """
    Memory-mapped HDF5 dataset supporting multiple H5 files with concatenated indexing.

    Efficiently loads data from disk without loading everything into RAM using HDF5's
    chunked storage and OS page caching. Supports multiple datasets, each with its
    own keypoint count; poses are zero-padded to the max keypoint count across
    datasets so they can be stacked into a batch, and each sample carries a
    dataset_id so a downstream model can route it to the right decoder head.

    Example:
        # Single-agent dataset (pose.h5) plus a social dataset backed by
        # pose_a1.h5 + pose_a2.h5 (both agents share one dataset_id)
        dataset = RodentDiskDataset(
            data_path='/path/to/data/',
            dataset_names=['wu_iclr25', 'sdannce'],
            train_val_test='train',
        )
    """

    def __init__(
        self,
        data_path: Union[str, dict],
        dataset_names: List[str],
        train_val_test: str,
        stride: int = None,
        use_pe_indices: bool = True,
        use_default_seg_len: bool = True,
        model_window: int = 51,
        min_interp_fps: Optional[int] = None,
        balanced_sampling: bool = True,
        virtual_rodent_x_splits: List[str] = ["base"],
        use_speed_filtered_inds: bool = False,
        scale_body: bool = False,
        pose_dtype: torch.dtype = torch.float32,
    ):
        """
        Parameters
        ----------
        data_path : str
            Root directory containing one subdirectory per dataset name.
        dataset_names : List[str]
            Dataset names to load (e.g. "sdannce", "pairr24m", "wu_iclr25", ...).
            Social datasets ("sdannce", "pairr24m") are backed by three H5
            files: pose_a1.h5/pose_a2.h5 (paired, sessions where both agents
            were tracked -- both agents share a single dataset_id and decoder
            head since they come from the same underlying dataset) and
            pose_single.h5 (sessions where only one agent was ever tracked).
        train_val_test : str
            Split to load ("train", "val", or "test").
        pose_dtype : torch.dtype
            dtype "pose" is cast to in __getitem__. Source H5 files are a mix
            of float32/float64 on disk; this normalizes to a single dtype.
        model_window, min_interp_fps : int, int or None
            Same model.window / train.augmentations.fps.min_fps values used
            to generate the "_2s" splits (befound.data.constants.get_window_size
            is a pure function of these plus each dataset's fps) -- needed to
            reconstruct each dataset's window size when the loaded indices
            file stores only a window's center frame rather than every frame
            in it. Ignored for any dataset whose indices file is still in the
            legacy full-window format (2D, window size read off its own shape).
        balanced_sampling : bool
            When multiple datasets are loaded and the DataLoader is built with
            shuffle=True, get_sampler(shuffle) returns a BalancedDatasetSampler
            so each dataset contributes ~equally per batch in expectation
            (sampling with replacement -- not every window is guaranteed to
            appear, and some may repeat, within a given epoch). Set this False
            for regular indexing instead: get_sampler returns None, so the
            DataLoader's own shuffle argument controls order and every window
            is seen exactly once per epoch, in proportion to each dataset's
            actual size.
        """
        self.pose_dtype = pose_dtype
        self.model_window = model_window
        self.min_interp_fps = min_interp_fps
        self.balanced_sampling = balanced_sampling
        self.use_speed_filtered_inds = use_speed_filtered_inds
        self.virtual_rodent_x_splits = virtual_rodent_x_splits
        self.dataset_names = list(
            dict.fromkeys(dataset_names)
        )  # de-duped, order-preserving
        self.dataset_name_to_idx = {
            name: i for i, name in enumerate(self.dataset_names)
        }

        if use_pe_indices:
            self.pe_indices = {}
        else:
            self.pe_indices = None

        self._init_pose_files(
            data_path, dataset_names, use_pe_indices, use_default_seg_len
        )

        if scale_body:
            self.body_scale = {}
            for name in dataset_names:
                if name == "sdannce":
                    self.body_scale[name] = BODY_SCALE[name]*SEQUENCE_SCALE
                elif name == "virtual_rodent":
                    if len(self.virtual_rodent_x_splits) == 3:
                        self.body_scale[name] = BODY_SCALE[name]*SEQUENCE_SCALE
                    else:
                        self.body_scale[name] = 0
                        n_sessions = 0
                        for x_split in self.virtual_rodent_x_splits:
                            n_sessions += BODY_SCALE[f"{name}_{x_split}"][1]
                            self.body_scale[name] += BODY_SCALE[f"{name}_{x_split}"][0] * BODY_SCALE[f"{name}_{x_split}"][1]
                        self.body_scale[name] /= n_sessions
                        self.body_scale[name] *= SEQUENCE_SCALE
                else:
                    self.body_scale[name] = BODY_SCALE[name]*SEQUENCE_SCALE

        else:
            self.body_scale = None

        if self.body_scale is not None and self.default_seg_len is not None:
            # Pre-normalize the canonical segment-length magnitudes by this
            # same body_scale, so get_default_offsets() returns offsets
            # already in the normalized units __getitem__ returns pose in --
            # callers (get_batch_loss/get_batch_loss_per_dataset) never need
            # to touch body_scale themselves.
            for name in self.dataset_names:
                self.default_seg_len[name] = (
                    self.default_seg_len[name] / self.body_scale[name]
                )

        self._init_indices(
            data_path, train_val_test, stride, model_window, min_interp_fps
        )

    def _init_pose_files(
        self, data_path, dataset_names, use_pe_indices, use_default_seg_len
    ):
        """Locate each dataset's h5 file(s), load its skeleton config
        (kinematic tree, offsets, PE indices, default segment lengths), and
        open every h5 file -- populates h5_paths/h5_key_to_dataset_name/
        dataset_to_h5_keys/kinematic_tree/skeleton_offsets/default_seg_len/
        pe_indices plus the opened h5_file_handles/h5_datasets/h5_ids/
        n_keypts/max_n_keypts/dataset_idx_to_n_keypts. body_scale is set by
        __init__ itself, after this call; window indices (self.indices etc.)
        are handled separately by _init_indices, once these h5_keys exist.
        """
        # Normalize to {h5_key -> (path, base_dataset_name)}
        h5_paths = {}
        h5_key_to_dataset_name = {}
        self.dataset_to_h5_keys = {}
        self.kinematic_tree = {}
        self.skeleton_offsets = {}
        self.data_path = data_path
        if use_default_seg_len:
            self.default_seg_len = {}
        else:
            self.default_seg_len = None

        for name in dataset_names:
            if name in ("pairr24m", "sdannce"):
                # Both are backed by pose_a1.h5/pose_a2.h5 (paired "social"
                # sessions, shared split) plus pose_single.h5 (sessions where
                # only one agent was ever tracked, independent split) -- see
                # ceph/data/pairr24m/code/split_pose_single.py for how
                # pairr24m's pose_single.h5 was carved out of pose_a1/a2.
                h5_paths[name + "_a1"] = data_path + name + "/pose_a1.h5"
                h5_paths[name + "_a2"] = data_path + name + "/pose_a2.h5"
                h5_paths[name + "_single"] = data_path + name + "/pose_single.h5"
                h5_key_to_dataset_name[name + "_a1"] = name
                h5_key_to_dataset_name[name + "_a2"] = name
                h5_key_to_dataset_name[name + "_single"] = name
                self.dataset_to_h5_keys[name] = [
                    name + "_a1",
                    name + "_a2",
                    name + "_single",
                ]

            elif name == "wu_iclr25":
                h5_paths[name] = data_path + name + "/parkinsons/pose.h5"
                h5_key_to_dataset_name[name] = name
                self.dataset_to_h5_keys[name] = [name]
            elif name == "virtual_rodent":
                self.dataset_to_h5_keys[name] = []
                for x_split in self.virtual_rodent_x_splits:
                    h5_paths[f"{name}_{x_split}"] = (
                        f"{data_path}{name}/pose_{x_split}.h5"
                    )
                    h5_key_to_dataset_name[f"{name}_{x_split}"] = name
                    self.dataset_to_h5_keys[name].append(f"{name}_{x_split}")
            else:
                h5_paths[name] = data_path + name + "/pose.h5"
                h5_key_to_dataset_name[name] = name
                self.dataset_to_h5_keys[name] = [name]

            skeleton_configs = read.config(data_path + name + "/skeleton.yaml")
            self.kinematic_tree[name] = skeleton_configs["KINEMATIC_TREE"]
            self.skeleton_offsets[name] = torch.tensor(skeleton_configs["OFFSET"], dtype=torch.float32)

            if use_pe_indices:
                self.pe_indices[name] = torch.tensor(skeleton_configs["PE_INDICES"], dtype=torch.long)

            if use_default_seg_len:
                self.default_seg_len[name] = torch.tensor(OFFSETS_3D[name], dtype=torch.float32)

        self.h5_paths = h5_paths
        self.h5_key_to_dataset_name = h5_key_to_dataset_name
        self.h5_file_handles = {}
        self.h5_datasets = {}
        self.h5_ids = {}
        self.n_keypts = {}  # base dataset_name -> n_keypts

        # Open all H5 files and cache memory-mapped references
        for h5_key, path in self.h5_paths.items():
            try:
                h5_file = h5py.File(path, "r")
                self.h5_file_handles[h5_key] = h5_file

                # Get memory-mapped reference to dataset
                self.h5_datasets[h5_key] = h5_file[
                    "pose"
                ]  # Assuming 'pose' is the key for pose data
                self.h5_ids[h5_key] = h5_file["ids"]

                dataset_name = h5_key_to_dataset_name[h5_key]
                self.n_keypts[dataset_name] = len(h5_file["pose"][0])

            except Exception as e:
                # Clean up already-opened files
                for h5f in self.h5_file_handles.values():
                    h5f.close()
                raise RuntimeError(f"Failed to open H5 file {path}: {e}")

        self.max_n_keypts = max(self.n_keypts.values())
        # dataset_idx -> n_keypts, matching the enumeration order used for dataset_id
        self.dataset_idx_to_n_keypts = {
            idx: self.n_keypts[name] for idx, name in enumerate(self.dataset_names)
        }

    def _init_indices(self, data_path, train_val_test, stride, model_window, min_interp_fps):
        """Load each h5_key's window indices for this split, derive its
        window size, and flatten every h5_key's windows into one global,
        indexable sample list (self._entries/_entry_dataset_idx). Depends on
        self.h5_paths/h5_key_to_dataset_name, so must run after
        _init_pose_files.
        """
        # Load window indices per H5 file (frame indices local to that file).
        # Two on-disk formats are supported: legacy full windows (2D,
        # (n_windows, window_size), window_size read off the array's own
        # shape) and the newer center-only format (1D, (n_windows,) -- just
        # each window's center frame; window_size instead derived via
        # get_window_size, the same pure function of model_window /
        # min_interp_fps / each dataset's fps that generated it, so it's
        # still never passed between the generator and this loader directly).
        self.indices = {}
        self.window_size = {}
        filtered_append = "_filtered_offsets" if self.use_speed_filtered_inds else ""
        for h5_key in self.h5_paths.keys():
            base_name = self.h5_key_to_dataset_name[h5_key]
            if h5_key.startswith("pairr24m") or h5_key.startswith("sdannce"):
                if h5_key.endswith("_a1") or h5_key.endswith("_a2"):
                    # a1/a2 share one split, saved with a "_social" suffix
                    self.indices[h5_key] = np.load(
                        f"{data_path}{base_name}/{train_val_test}_inds_social_2s{filtered_append}.npy"
                    ).astype(np.int32)[:: stride]
                elif h5_key.endswith("_single"):
                    self.indices[h5_key] = np.load(
                        f"{data_path}{base_name}/{train_val_test}_inds_single_2s{filtered_append}.npy"
                    ).astype(np.int32)[:: stride]
            elif base_name == "wu_iclr25":
                self.indices[h5_key] = np.load(
                    f"{data_path}{base_name}/parkinsons/{train_val_test}/inds_2s{filtered_append}.npy"
                ).astype(np.int32)[:: stride]
            elif base_name == "virtual_rodent":
                if h5_key.endswith("_base"):
                    self.indices[h5_key] = np.load(
                        f"{data_path}{base_name}/{train_val_test}_inds_2s{filtered_append}.npy"
                    ).astype(np.int32)[:: stride]
                else:
                    x_split = h5_key.split("_")[-1]
                    self.indices[h5_key] = np.load(
                        f"{data_path}{base_name}/{train_val_test}_inds_{x_split}_2s{filtered_append}.npy"
                    ).astype(np.int32)[:: stride]
            else:
                self.indices[h5_key] = np.load(
                    f"{data_path}{base_name}/{train_val_test}_inds_2s{filtered_append}.npy"
                ).astype(np.int32)[:: stride]

            if self.indices[h5_key].ndim == 2:
                self.window_size[base_name] = self.indices[h5_key].shape[-1]
            else:
                self.window_size[base_name] = get_window_size(
                    base_name, model_window, min_interp_fps
                )

            print(
                f"Loading {train_val_test} indices for {h5_key}: {len(self.indices[h5_key])} windows"
            )

        self.max_window_size = max(self.window_size.values())
        # dataset_idx -> window_size, matching the enumeration order used for dataset_id
        self.window_size_actual = {
            idx: self.window_size[name] for idx, name in enumerate(self.dataset_names)
        }

        # Flatten per-file windows into one global, indexable sample list.
        # Each entry is (h5_key, local_window_idx); pose_a1/pose_a2 windows
        # are separate samples but share dataset_id (same base dataset).
        self._entries = []
        self._entry_dataset_idx = []
        for h5_key in self.h5_paths.keys():
            dataset_idx = self.dataset_name_to_idx[self.h5_key_to_dataset_name[h5_key]]
            n_windows = len(self.indices[h5_key])
            self._entries.extend((h5_key, i) for i in range(n_windows))
            self._entry_dataset_idx.extend([dataset_idx] * n_windows)

    def __len__(self) -> int:
        """Return number of windows across all datasets/files"""
        return len(self._entries)

    def _full_window_indices(self, indices: np.ndarray, base_name: str) -> np.ndarray:
        """Expand `indices` to full window frame-index array(s), regardless of
        whether the on-disk format stores full windows or just each window's
        center frame. Handles both a batch (indices.ndim == 2: full windows,
        returned as-is; ndim == 1 with length != window_size: a batch of
        center indices, expanded) and a single sample already pulled out of
        that batch (ndim == 1 with length == window_size: one full window,
        returned as-is; ndim == 0: one center index, expanded) -- the same
        window_size-vs-not-window_size length check distinguishes "already a
        full window" from "a batch of centers" in both cases."""
        indices = np.asarray(indices)
        window_size = self.window_size[base_name]
        if indices.ndim == 2 or (
            indices.ndim == 1 and indices.shape[-1] == window_size
        ):
            return indices
        half = window_size // 2
        offsets = np.arange(-half, half + 1)
        return indices[..., None] + offsets

    def get_default_offsets(self, device) -> Optional[Dict[str, torch.Tensor]]:
        """Combined (n_keypts, 3) canonical offsets (magnitude x direction)
        per dataset, used by get_batch_loss/get_batch_loss_per_dataset to
        reconstruct pose_hat in the same canonical skeleton space
        target_pose was built in. None if this dataset has no
        default_seg_len (use_default_seg_len=False)."""
        if not self.default_seg_len:
            return None
        return {
            k: (self.default_seg_len[k][:, None] * self.skeleton_offsets[k]).to(device)
            for k in self.dataset_names
        }

    def get_sampler(self, shuffle: bool) -> Optional["BalancedDatasetSampler"]:
        """
        Sampler to pass to the DataLoader, or None.

        If self.balanced_sampling and shuffle and more than one dataset is
        loaded: a BalancedDatasetSampler so each dataset contributes ~equally
        per batch in expectation, sampling with replacement -- not a strict
        single pass over the data. (Not a WeightedRandomSampler: that calls
        torch.multinomial, which hard-caps the number of categories --
        i.e. len(self) here -- at 2**24, far below this class's typical
        window counts.)

        Otherwise: None, for regular indexing -- pass this straight to
        DataLoader(sampler=...) alongside shuffle=shuffle (unchanged); with no
        sampler, the DataLoader's own shuffle argument determines order and
        every window is returned exactly once per epoch.
        """
        if not shuffle or not self.balanced_sampling or len(self.dataset_names) <= 1:
            return None
        return BalancedDatasetSampler(
            self._entry_dataset_idx, len(self.dataset_names), len(self)
        )

    def __getitem__(self, idx: int) -> dict:
        """
        Get a sample by global window index.

        Pose is zero-padded along the time axis to self.max_window_size and
        along the keypoint axis to self.max_n_keypts so that samples from
        datasets with different window sizes and/or keypoint counts can be
        stacked by the default collate_fn. dataset_id/dataset_name identify
        which decoder head the sample should be routed to; pose_a1/pose_a2
        windows from a social dataset are separate samples that share the
        same dataset_id.

        Returns
        -------
        dict with keys:
            pose : torch.Tensor, shape (max_window_size, max_n_keypts, 3), self.pose_dtype
            ids : torch.Tensor, scalar, int16 (matches on-disk H5 "ids" dtype)
            dataset_id : torch.Tensor, scalar, int8 (small enumeration of datasets)
            dataset_name : str
        """
        h5_key, local_window_idx = self._entries[idx]
        if len(self.dataset_names) != 1:
            dataset_idx = self._entry_dataset_idx[idx]
        # dataset_name = self.dataset_names[dataset_idx]

        # A single sample's raw stored value is either a full window (2D
        # on-disk format) or just its center frame (1D format) -- expand the
        # latter cheaply per-sample here (_full_window_indices also handles
        # the batched case, used instead for the one-time
        # remove_speed_outliers setup in __init__).
        base_name = self.h5_key_to_dataset_name[h5_key]
        raw = self.indices[h5_key][local_window_idx]
        window_inds = self._full_window_indices(raw, base_name)

        pose = self.h5_datasets[h5_key][window_inds]
        # Center frame of this sample's own window, not a fixed global window
        # size -- window length now varies per dataset.
        ids = self.h5_ids[h5_key][window_inds[len(window_inds) // 2]]

        pose = torch.from_numpy(pose).to(self.pose_dtype)
        if self.body_scale is not None:
            pose = pose / self.body_scale[base_name]
        n_frames, n_keypts = pose.shape[0], pose.shape[1]

        if n_frames < self.max_window_size:
            pose = torch.nn.functional.pad(
                pose, (0, 0, 0, 0, 0, self.max_window_size - n_frames)
            )
        if n_keypts < self.max_n_keypts:
            pose = torch.nn.functional.pad(
                pose, (0, 0, 0, self.max_n_keypts - n_keypts)
            )

        data = {
            "pose": pose,
            "ids": torch.from_numpy(np.asarray(ids)).to(torch.int16),
        }
        if self.pe_indices is not None:
            pe_indices = self.pe_indices[self.h5_key_to_dataset_name[h5_key]]
            if n_keypts < self.max_n_keypts:
                # Pad with 6, the reserved "no region" class (PE_INDICES only uses
                # 0-5) so ChannelInvariantEncoder's 7-slot embedding stays in range;
                # per-dataset processing slices back to dataset_idx_to_n_keypts regardless.
                pe_indices = torch.nn.functional.pad(
                    pe_indices, (0, self.max_n_keypts - n_keypts), value=6
                )
            data["pe_indices"] = pe_indices
        if len(self.dataset_names) != 1:
            data["dataset_id"] = torch.tensor(dataset_idx, dtype=torch.int8)
        return data

    def __del__(self):
        """Clean up by closing all H5 files"""
        for h5_file in self.h5_file_handles.values():
            try:
                h5_file.close()
            except:
                pass

    def close(self):
        """Explicitly close all H5 files"""
        self.__del__()
