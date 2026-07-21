from neuroposelib import read
import numpy as np
import befound.data.quaternion as qtn
from befound.data.constants import OFFSETS_3D_BOX, OFFSETS_3D_SUM_BOX, OFFSETS_3D_PAIRR24M, OFFSETS_3D_SUM_PAIRR24M
from typing import Optional, Type, Union, List
from torch.utils.data import Dataset
import torch
from numpy.lib.stride_tricks import sliding_window_view
from tqdm import trange
from neuroposelib import read
import numpy as np
from typing import List
import torch
from torch.utils.data import DataLoader
import h5py
import pandas as pd
from befound.data.data import (
    inv_kin,
    fwd_kin_cont6d_torch,
    get_segment_len,
    get_speed_parts,
    get_frame_yaw,
    get_angle2D,
    get_window_indices,
    get_speed_outliers,
)

def _preprocess_pose_to_features(
    pose,
    window,
    yaw_root_i=0,
    yaw_front_i=1,
    skeleton_config=None,
    data_keys=None,
    direction_process="midfwd",
    use_default_offsets=False,
    offsets_default=None,
):
    """
    Common pose preprocessing pipeline: yaw, x6d, offsets, root, target_pose.
    Used by box, pairr24m, and mouse_44_ephys loaders.

    Parameters
    ----------
    pose : np.ndarray
        Shape (n_samples, window, n_keypoints, 3)
    window : int
        Window length (used to find middle frame)
    yaw_root_i, yaw_front_i : int
        Keypoint indices for yaw computation
    skeleton_config : dict
        Contains KINEMATIC_TREE and OFFSET
    data_keys : List[str]
        Which features to compute
    direction_process : str
        "midfwd" or "x360" for direction preprocessing
    use_default_offsets : bool
        Whether to use pre-computed default offsets
    offsets_default : np.ndarray, optional
        Default offsets to use if use_default_offsets=True

    Returns
    -------
    dict
        Dictionary with keys from data_keys: x6d, root, offsets, heading, target_pose, etc.
    """
    if data_keys is None:
        data_keys = []

    data = {}

    # Get yaw of the segment for central frame in all windows
    yaw = get_frame_yaw(pose[:, window // 2, ...], yaw_root_i, yaw_front_i)[..., None]

    # Convert to 2D representation using sin and cos of yaw
    if "heading" in data_keys:
        data["heading"] = get_angle2D(yaw)

    # Root processing
    if ("root" or "x6d") in data_keys:
        root = pose[..., 0, :].copy()  # (n_samples, window, 3)
        if direction_process in ["midfwd", "x360"]:
            root_center = np.zeros(root.shape)
            root_center[..., [0, 1]] = root[:, window // 2, [0, 1]][:, None, :]
            root -= root_center

    # Inverse kinematics to continuous 6D
    if "x6d" in data_keys and skeleton_config is not None:
        print("Applying inverse kinematics ...")
        local_qtn = inv_kin(
            pose.reshape((-1,) + pose.shape[-2:]),
            skeleton_config["KINEMATIC_TREE"],
            np.array(skeleton_config["OFFSET"]),
            forward_indices=[1, 0],
        ).reshape(pose.shape[:-1] + (-1,))

        if direction_process == "midfwd":
            fwd_qtn = np.zeros((len(yaw), 4))
            fwd_qtn[:, [-1, 0]] = get_angle2D(yaw / 2)
            fwd_qtn = np.repeat(fwd_qtn[:, None, :], window, axis=1)
            local_qtn[..., 0, :] = qtn.qmul_np(fwd_qtn, local_qtn[..., 0, :])

            if "root" in data_keys:
                root = qtn.qrot_np(fwd_qtn, root)

        data["x6d"] = qtn.quaternion_to_cont6d_np(local_qtn)

    # Offsets (segment lengths)
    if "offsets" in data_keys and skeleton_config is not None:
        if use_default_offsets:
            data["offsets"] = offsets_default[:, None] * np.array(
                skeleton_config["OFFSET"], dtype=np.float32
            )
        else:
            data["offsets"] = get_segment_len(
                pose.reshape((-1,) + pose.shape[-2:]),
                skeleton_config["KINEMATIC_TREE"],
                np.array(skeleton_config["OFFSET"]),
            ).reshape(pose.shape)

    # Root positions
    if "root" in data_keys:
        data["root"] = root
        frame_dim_inds = tuple(range(len(root.shape) - 1))
        print("Root Maxes: {}".format(root.max(axis=frame_dim_inds)))
        print("Root Mins: {}".format(root.min(axis=frame_dim_inds)))

    # Target pose (forward kinematics from x6d)
    if "target_pose" in data_keys and "x6d" in data_keys and skeleton_config is not None:
        reshaped_x6d = data["x6d"].reshape((-1,) + data["x6d"].shape[-2:])
        if use_default_offsets:
            offsets = torch.from_numpy(offsets_default).float()
        else:
            offsets = torch.from_numpy(
                data["offsets"].reshape(reshaped_x6d.shape[:2] + (-1,))
            ).float()
        data["target_pose"] = fwd_kin_cont6d_torch(
            torch.from_numpy(reshaped_x6d).float(),
            skeleton_config["KINEMATIC_TREE"],
            offsets,
            root_pos=torch.zeros(reshaped_x6d.shape[0], 3),
            do_root_R=True,
            eps=1e-8,
        ).reshape(data["x6d"].shape[:-1] + (3,))

    # Convert numpy arrays to torch tensors
    data = {k: torch.tensor(v, dtype=torch.float32) for k, v in data.items()}

    return data


def preprocess_box_data(
    data_path: str,
    skeleton_config: dict,
    dataset: str,
    window: int,
    train_val_test: str = "train",
    stride: int = 2,
    data_keys: List[str] = ["x6d", "root", "offsets"],
    speed_threshold: Optional[float] = 2.25,
    direction_process: str = "midfwd",
    use_default_offsets: bool = False,
):
    """Prepare and save all data preprocessing for SC-VAE model training, validation, and testing

    Parameters
    ----------
    data_path : str
        Path to folder with datasets
    skeleton_config : dict
        Configuration file denoting the structure of the skeleton
    dataset : str
        Which dataset to prepare (i.e., "4_mice", "parkinsons")
    data_keys : List[str], optional
        Keys of data to save, by default ["x6d", "root", "offsets"]
    speed_threshold : Optional[float], optional
        Action segments with greater average speed will be filtered out, by default 2.25
    direction_process : str, optional
        Preprocess pose sequences such that the animals pass through the origin at the middle frame from
        any direction ("x360") or only in the x+ direction ("midfwd"), by default "midfwd"
    use_default_offsets : bool, optional
        Whether to use default segment lengths for offsets, by default False

    Returns
    -------
    data
        Dictionary with key-value pairs associated with `data_keys`
    """
    n_ids = 72 if "parkinsons" in dataset else 4
    print("Calculating dataset: {}".format(dataset))
    dataset_name = "parkinsons" if dataset == "parkinsons_healthy" else dataset
    if train_val_test in [None, "full"]:
        pose, ids = read.pose_h5("{}{}/pose.h5".format(data_path, dataset_name))
        window_inds = get_window_indices(ids, stride, window)
        pose = pose[window_inds]
        ids = ids[window_inds][:, window // 2]
    else:
        pose = np.load(
            "{}{}/{}/pose.npy".format(data_path, dataset_name, train_val_test)
        )
        # pose = pose[..., 51 // 2, :, :]
        ids = np.repeat(np.arange(n_ids), len(pose) // n_ids)

    if dataset == "parkinsons_healthy":
        pose = pose[ids < 36, ...]
        ids = ids[ids < 36]

    # Filter out bad tracking using speed threshold
    if speed_threshold is not None:
        outlier_frames = get_speed_outliers(pose, speed_threshold)
        pose = np.delete(pose, outlier_frames, 0)
        ids = np.delete(ids, outlier_frames, 0)

    data_len = len(pose)
    data = {"raw_pose": pose}
    # Calculate the speed representation
    if "avg_speed_3d" in data_keys:
        speed = get_speed_parts(
            pose=pose,
            parts=[
                [0, 1, 2, 3, 4, 5],  # spine and head
                [1, 6, 7, 8, 9, 10, 11],  # arms from front spine
                [5, 12, 13, 14, 15, 16, 17],  # left legs from back spine
            ],
        )

        data["avg_speed_3d"] = np.concatenate(
            [speed[:, :2], speed[:, 2:].mean(axis=-1, keepdims=True)], axis=-1
        )

    # Compute pose features (yaw, heading, x6d, offsets, root, target_pose)
    features = _preprocess_pose_to_features(
        pose,
        window=window,
        yaw_root_i=0,
        yaw_front_i=1,
        skeleton_config=skeleton_config,
        data_keys=data_keys,
        direction_process=direction_process,
        use_default_offsets=use_default_offsets,
        offsets_default=OFFSETS_3D_BOX if use_default_offsets else None,
    )
    data.update(features)

    # Get animal IDs
    if "ids" in data_keys:
        data["ids"] = torch.tensor(ids, dtype=torch.int16)

    if "processed_pose" in data_keys:
        reshaped_x6d = data["x6d"].reshape((-1,) + data["x6d"].shape[-2:])
        if use_default_offsets:
            offsets = data["offsets"]
        else:
            offsets = data["offsets"].reshape(reshaped_x6d.shape[:2] + (-1,))
        data["target_pose"] = fwd_kin_cont6d_torch(
            reshaped_x6d,
            skeleton_config["KINEMATIC_TREE"],
            offsets,
            root_pos=torch.zeros(reshaped_x6d.shape[0], 3),
            do_root_R=True,
            eps=1e-8,
        ).reshape(data["x6d"].shape[:-1] + (3,))

    if "target_pose" in data_keys:
        # Target pose root does not move
        reshaped_x6d = data["x6d"].reshape((-1,) + data["x6d"].shape[-2:])
        if use_default_offsets:
            offsets = data["offsets"]
        else:
            offsets = data["offsets"].reshape(reshaped_x6d.shape[:2] + (-1,))
        data["target_pose"] = fwd_kin_cont6d_torch(
            reshaped_x6d,
            skeleton_config["KINEMATIC_TREE"],
            offsets,
            root_pos=torch.zeros(reshaped_x6d.shape[0], 3),
            do_root_R=True,
            eps=1e-8,
        ).reshape(data["x6d"].shape[:-1] + (3,))

    for k, v in data.items():
        try:
            assert len(v) == data_len
        except:
            assert (len(v) == pose.shape[-2]) and (k == "offsets")

    return data


def preprocess_pairr24m_data(
    data_path: str,
    skeleton_config: dict,
    dataset: str,
    window: int,
    train_val_test: str = "train",
    stride: int = 1,
    data_keys: List[str] = ["x6d", "root", "offsets"],
    direction_process: str = "midfwd",
    use_default_offsets: bool = False,
    get_social_paired: bool = False,
    demo: bool = False,
):
    """Prepare and save all data preprocessing for SC-VAE model training, validation, and testing

    Parameters
    ----------
    data_path : str
        Path to folder with datasets
    skeleton_config : dict
        Configuration file denoting the structure of the skeleton
    dataset : str
        Which dataset to prepare (i.e., "4_mice", "parkinsons")
    data_keys : List[str], optional
        Keys of data to save, by default ["x6d", "root", "offsets"]
    direction_process : str, optional
        Preprocess pose sequences such that the animals pass through the origin at the middle frame from
        any direction ("x360") or only in the x+ direction ("midfwd"), by default "midfwd"
    use_default_offsets : bool, optional
        Whether to use default segment lengths for offsets, by default False

    Returns
    -------
    data
        Dictionary with key-value pairs associated with `data_keys`
    """

    meta = pd.read_csv(data_path + "metadata.csv")

    labels_list = [
        "goodFrame_an1",
        "goodFrame_an2",
        "interactionCat",
        "behaviorFine_an1",
        "behaviorFine_an2",
        "behaviorCoarse_an1",
        "behaviorCoarse_an2",
    ]
    pose_a1, ids = read.pose_h5(data_path + "pose_a1.h5")
    pose_a2, _ = read.pose_h5(data_path + "pose_a2.h5")
    split_inds = np.load(data_path + "{}_inds.npy".format(train_val_test))
    pose_a1 = pose_a1.astype(np.float32)
    pose_a2 = pose_a2.astype(np.float32)
    split_inds = split_inds.astype(np.int32)
    ids = ids.astype(np.int16)

    curr_id = 0
    split_ids = []
    for i in np.unique(ids):
        split_inds_i = split_inds[ids[split_inds] == i]
        n_consecutive = np.where(
            np.diff(split_inds_i, prepend=split_inds_i[0] - 1) != 1
        )[0]
        n_consecutive = np.diff(n_consecutive, prepend=0, append=len(split_inds_i))
        split_ids += [np.repeat(np.arange(len(n_consecutive)) + curr_id, n_consecutive)]
        curr_id += len(n_consecutive)

    split_ids = np.concatenate(split_ids)
    # labels_dict = {k: [] for k in labels_list}
    # for i, f_path in enumerate(meta["FolderPath"]):
    #     markers_csv = pd.read_csv(data_path + f_path + "/markerDataset.csv")
    #     # for label in labels_list:
    #     #     labels_dict[label].append(markers[label].to_numpy())
    # labels_dict = {
    #     k: torch.from_numpy(np.concatenate(v, axis=0)) for k, v in labels_dict.items()
    # }

    # for label in ["goodFrame", "behaviorFine", "behaviorCoarse"]:
    #     snake_case = re.sub(r"(?<!^)(?=[A-Z])", "_", label).lower()
    #     if snake_case in data_keys:
    #         labels_dict[re.sub(r"(?<!^)(?=[A-Z])", "_", label).lower()] = torch.vstack(
    #             (labels_dict[label + "_an1"], labels_dict[label + "_an2"])
    #         )
    #         del labels_dict[label + "_an1"]
    #         del labels_dict[label + "_an2"]

    window_inds = get_window_indices(split_ids, stride, window)

    if demo:
        window_inds = np.concatenate(
            [
                window_inds[:20000, ...],
                window_inds[len(window_inds) // 2 : len(window_inds) // 2 + 20000],
            ],
            axis=0,
        )
    ids = ids[split_inds[window_inds]][:, window // 2]
    pose_a1 = pose_a1[split_inds[window_inds]]
    pose_a2 = pose_a2[split_inds[window_inds]]

    # pose = pose[window_inds, ...].astype(np.float32)
    # ids = ids[window_inds][:, window // 2]

    pose = np.concatenate([pose_a1, pose_a2], axis=0)
    ids = np.concatenate([ids, ids + ids.max() + 1])

    print("Calculating dataset: {}".format(dataset))

    # Filter out bad tracking given nan lables
    nan_frames = np.unique(np.where(np.isnan(pose))[0])
    pose = np.delete(pose, nan_frames, axis=0)
    ids = np.delete(ids, nan_frames, axis=0)

    data_len = len(pose)
    data = {}  # "raw_pose": pose}

    # Compute pose features (yaw, heading, x6d, offsets, root, target_pose)
    features = _preprocess_pose_to_features(
        pose,
        window=window,
        yaw_root_i=0,
        yaw_front_i=1,
        skeleton_config=skeleton_config,
        data_keys=data_keys,
        direction_process=direction_process,
        use_default_offsets=use_default_offsets,
        offsets_default=OFFSETS_3D_PAIRR24M if use_default_offsets else None,
    )
    data.update(features)
    # if get_paired:
    #     data = {
    #         k: v.reshape((2, -1, *v.shape[1:])) if k != "offsets" else v
    #         for k, v in data.items()
    #     }
    # data.update(labels_dict)

    # Get animal IDs
    if "ids" in data_keys:
        data["ids"] = torch.tensor(ids, dtype=torch.int16)

    for k, v in data.items():
        try:
            assert len(v) == data_len
        except:
            assert (len(v) == pose.shape[-2]) and (k == "offsets")

    return data


def preprocess_mouse44_ephys_data(
    data_path: str,
    split_indices_path: str,
    skeleton_config: dict,
    train_val_test: str = "train",
    window: int = 51,
    data_keys: List[str] = ["x6d", "root", "offsets"],
    direction_process: str = "midfwd",
    use_default_offsets: bool = False,
    keypoint_reorder: List[int] = None,
):
    """
    Load and preprocess mouse44_ephys dataset using pre-computed train/val/test splits.

    Parameters
    ----------
    data_path : str
        Path to folder with .mat files
    split_indices_path : str
        Path to directory with train_inds.npy, val_inds.npy, test_inds.npy
    skeleton_config : dict
        Skeleton config with KINEMATIC_TREE and OFFSET
    train_val_test : str
        Which split ("train", "val", "test")
    window : int
        Sliding window length
    data_keys : List[str]
        Features to compute (x6d, root, offsets, heading, etc.)
    direction_process : str
        "midfwd" or "x360"
    use_default_offsets : bool
        Use pre-computed default offsets
    keypoint_reorder : List[int], optional
        Keypoint reordering indices

    Returns
    -------
    data : dict
        Torch tensors for all requested data_keys
    """
    import hdf5storage
    from pathlib import Path

    # Load all .mat files
    mat_files = sorted(Path(data_path).glob("*.mat"))
    pose_dict = {}
    for mat_file in mat_files:
        session_data = hdf5storage.loadmat(str(mat_file))
        pose = session_data["keypoints"].astype(np.float32)
        if keypoint_reorder is not None:
            pose = pose[:, keypoint_reorder, :]
        pose_dict[mat_file.name] = pose

    # Concatenate pose
    pose_full = np.concatenate([pose_dict[s] for s in pose_dict.keys()], axis=0)

    # Load split indices
    split_path = Path(split_indices_path)
    if train_val_test == "train":
        inds = np.load(split_path / "train_inds.npy")
    elif train_val_test == "val":
        inds = np.load(split_path / "val_inds.npy")
    elif train_val_test == "test":
        inds = np.load(split_path / "test_inds.npy")
    else:
        raise ValueError(f"Unknown split: {train_val_test}")

    # Index windowed pose
    pose = pose_full[inds]  # (n_samples, window, n_keypoints, 3)

    data_len = len(pose)
    data = {"raw_pose": pose}

    # Compute features
    features = _preprocess_pose_to_features(
        pose,
        window=window,
        yaw_root_i=0,
        yaw_front_i=1,
        skeleton_config=skeleton_config,
        data_keys=data_keys,
        direction_process=direction_process,
        use_default_offsets=use_default_offsets,
        offsets_default=OFFSETS_3D_PAIRR24M if use_default_offsets else None,
    )
    data.update(features)

    # Validate all outputs have same length
    for k, v in data.items():
        try:
            assert len(v) == data_len
        except:
            assert (len(v) == pose.shape[-2]) and (k == "offsets")

    return data
