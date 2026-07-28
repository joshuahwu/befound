from neuroposelib import read, write
import numpy as np
import befound.data.quaternion as qtn
from befound.data.constants import OFFSETS_3D, OFFSETS_3D_SUM
from typing import Optional, Type, Union, List
from torch.utils.data import Dataset
import torch
from numpy.lib.stride_tricks import sliding_window_view
from tqdm import trange
import numpy as np
from typing import List
import torch
from torch.utils.data import DataLoader
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
from pathlib import Path
import h5py

def _preprocess_pose_to_features(
    pose,
    skeleton_config=None,
    data_keys=None,
    direction_process="midfwd",
    use_default_offsets=False,
    offsets_default=None,
    **kwargs,
):
    """
    Common pose preprocessing pipeline: yaw, x6d, offsets, root, target_pose.
    Used by wu_iclr25, pairr24m, and mouse_44_ephys loaders.

    Parameters
    ----------
    pose : np.ndarray
        Shape (n_samples, window, n_keypoints, 3)
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
    window = pose.shape[-3]  # Ensure window matches pose shape
    if data_keys is None:
        data_keys = []

    data = {}
    # Get yaw of the segment for central frame in all windows
    yaw = get_frame_yaw(pose[:, window // 2, ...], 0, 1)[..., None]

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

    # Convert numpy arrays to torch tensors
    data = {k: torch.tensor(v, dtype=torch.float32) for k, v in data.items()}

    # Target pose (forward kinematics from x6d)
    if (
        "target_pose" in data_keys
        and "x6d" in data_keys
        and skeleton_config is not None
    ):
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


    return data


def preprocess_wu_iclr25_data(
    data_path: str,
    skeleton_config: dict,
    train_val_test: str = "train",
    data_keys: List[str] = ["x6d", "root", "offsets"],
    direction_process: str = "midfwd",
    use_default_offsets: bool = False,
    **kwargs,
):
    """Load pre-calculated features from wu_iclr25 datasets.

    Loads pre-computed x6d, root, offsets, etc. from H5 files.
    Handles ids, pd_label (for parkinsons), fluorescence, and discrete classes.

    Parameters
    ----------
    data_path : str
        Path to folder with datasets
    skeleton_config : dict
        Configuration file denoting the structure of the skeleton
    window : int
        Window size (used for validation)
    train_val_test : str
        Which split ("train", "val", "test")
    stride : int
        Stride (informational)
    data_keys : List[str]
        Features to load (x6d, root, offsets, heading, etc.)
    direction_process : str
        Direction processing mode (midfwd, x360)
    use_default_offsets : bool
        Whether to use default offsets

    Returns
    -------
    data : dict
        Dictionary with torch tensors for all requested data_keys
    """
    dataset_name = "parkinsons"
    dataset_path = "{}{}/{}/".format(data_path, dataset_name, train_val_test)

    # Handle x3d which requires additional keys
    if "x3d" in data_keys:
        if "x6d" not in data_keys:
            data_keys = list(data_keys) + ["x6d"]
        if "offsets" not in data_keys:
            data_keys = list(data_keys) + ["offsets"]
        if "root" not in data_keys:
            data_keys = list(data_keys) + ["root"]

    # Load pre-calculated features from H5 files
    data = {}
    for key in data_keys:
        if key in ["pd_label", "fluorescence", "x3d"]:
            continue
        elif key in ["ids", "heading", "avg_speed_3d", "raw_pose"]:
            file_path = "{}{}.h5".format(dataset_path, key)
        elif key == "offsets":
            if use_default_offsets:
                data["offsets"] = OFFSETS_3D["wu_iclr25"]
                data["offsets"] = data["offsets"][:, None] * np.array(
                    skeleton_config["OFFSET"], dtype=np.float32
                )
                continue
            else:
                file_path = "{}{}.h5".format(dataset_path, key)
        else:
            file_path = "{}{}_{}.h5".format(dataset_path, key, direction_process)

        print("Reading in {} from {}".format(key, file_path))
        hf = h5py.File(file_path, "r")
        data[key] = np.array(hf.get(key))
        hf.close()

    # Convert to torch tensors
    data = {k: torch.from_numpy(v) if isinstance(v, np.ndarray) else v for k, v in data.items()}

    data_len = len(data[list(data.keys())[0]])
    ids = data["ids"].numpy() if isinstance(data["ids"], torch.Tensor) else data["ids"]

    # Get animal IDs
    if "ids" in data_keys:
        data["ids"] = torch.tensor(ids, dtype=torch.int16)

    data["discrete_classes"] = {}
    # if data_config["dataset"] == "parkinsons":
    # Only if read in raw poses for the PD dataset
    # if not ((data_config["stride"] == 5) or (data_config["stride"] == 10)):
    if "pd_label" in data_keys:
        data["pd_label"] = torch.zeros((len(data["ids"]), 1)).long()
        data["pd_label"][data["ids"] >= 36] = 1
        data["discrete_classes"]["pd_label"] = torch.unique(
            data["pd_label"], sorted=True
        )

    if "fluorescence" in data_keys:
        meta = pd.read_csv(dataset_path + "metadata.csv")
        meta_by_frame = meta.iloc[data["ids"]]
        fluorescence = meta_by_frame["Fluorescence"].to_numpy()
        data["fluorescence"] = torch.tensor(fluorescence, dtype=torch.float32)

    data["a_ids"] = torch.tensor(ids, dtype=torch.int16)
    data["a_ids"][data["a_ids"] >= 36] = data["a_ids"][data["a_ids"] >= 36] - 36
    unique_ids = torch.unique(data["a_ids"])
    data["discrete_classes"]["a_ids"] = torch.arange(len(unique_ids)).long()

    # Validate all data has same length
    for k, v in data.items():
        if k == "discrete_classes":
            continue
        try:
            assert len(v) == data_len, f"Length mismatch for {k}: {len(v)} != {data_len}"
        except AssertionError as e:
            # offsets can have a different first dimension (n_keypoints,)
            if k == "offsets" and len(v.shape) == 2:
                assert v.shape[0] == len(skeleton_config["OFFSET"])
            else:
                raise e

    return data


def preprocess_pairr24m_data(
    data_path: str,
    skeleton_config: dict,
    window: int,
    train_val_test: str = "train",
    stride: int = 1,
    data_keys: List[str] = ["x6d", "root", "offsets"],
    direction_process: str = "midfwd",
    use_default_offsets: bool = False,
    demo: bool = False,
    **kwargs,
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

    # Filter out bad tracking given nan lables
    nan_frames = np.unique(np.where(np.isnan(pose))[0])
    pose = np.delete(pose, nan_frames, axis=0)
    ids = np.delete(ids, nan_frames, axis=0)

    data_len = len(pose)
    data = {}  # "raw_pose": pose}

    # Compute pose features (yaw, heading, x6d, offsets, root, target_pose)
    features = _preprocess_pose_to_features(
        pose,
        skeleton_config=skeleton_config,
        data_keys=data_keys,
        direction_process=direction_process,
        use_default_offsets=use_default_offsets,
        offsets_default=OFFSETS_3D["pairr24m"] if use_default_offsets else None,
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
    skeleton_config: dict,
    train_val_test: str = "train",
    stride: int = 1,
    data_keys: List[str] = ["x6d", "root", "offsets"],
    direction_process: str = "midfwd",
    use_default_offsets: bool = False,
    **kwargs,
):
    """
    Load and preprocess mouse44_ephys dataset using pre-computed train/val/test splits.

    Parameters
    ----------
    data_path : str
        Path to folder with .mat files
    skeleton_config : dict
        Skeleton config with KINEMATIC_TREE and OFFSET
    train_val_test : str
        Which split ("train", "val", "test")
    data_keys : List[str]
        Features to compute (x6d, root, offsets, heading, etc.)
    direction_process : str
        "midfwd" or "x360"
    use_default_offsets : bool
        Use pre-computed default offsets

    Returns
    -------
    data : dict
        Torch tensors for all requested data_keys
    """

    # Load all .mat files
    pose, ids = read.pose_h5(data_path + "pose.h5")  # (n_samples, n_keypoints, 3)

    # Load split indices
    if train_val_test == "train":
        inds = np.load(data_path + "train_inds.npy")
    elif train_val_test == "val":
        inds = np.load(data_path + "val_inds.npy")
    elif train_val_test == "test":
        inds = np.load(data_path + "test_inds.npy")
    else:
        raise ValueError(f"Unknown split: {train_val_test}")

    # Index windowed pose
    window = inds.shape[-1]
    pose = pose[inds[::stride]]  # (n_samples, window, n_keypoints, 3)
    ids = ids[inds[::stride, window//2]]

    data_len = len(pose)
    data = {}

    # Compute features
    features = _preprocess_pose_to_features(
        pose,
        skeleton_config=skeleton_config,
        data_keys=data_keys,
        direction_process=direction_process,
        use_default_offsets=use_default_offsets,
        offsets_default=OFFSETS_3D["mouse44_ephys"] if use_default_offsets else None,
    )
    data.update(features)
    # Get animal IDs
    if "ids" in data_keys:
        data["ids"] = torch.tensor(ids, dtype=torch.int16)
    # import pdb; pdb.set_trace()
    # samples = [50000, 2000, 124058]
    # vis.pose.grid3D(data["target_pose"][samples].numpy().reshape(-1, 44, 3), connectivity, frames = np.arange(len(samples))*51, centered=False, fps=35, N_FRAMES=51, VID_NAME="test.mp4", SAVE_ROOT="./")

    # Validate all outputs have same length
    for k, v in data.items():
        try:
            assert len(v) == data_len
        except:
            assert (len(v) == pose.shape[-2]) and (k == "offsets")

    return data


def train_val_test_split(
    pose_dict,
    window=51,
    stride=10,
    block_size=6000,
    train_frac=0.5,
    val_frac=0.25,
    test_frac=0.25,
    seed=0,
    save_dir=None,
    gap=25,
    pose_h5_path=None,
):

    """
    Split multi-session pose data into train/val/test by temporal blocks.

    Each session is chopped into fixed-size blocks, and whole blocks are randomly
    assigned to splits. Windows straddling block boundaries are dropped to prevent
    temporal leakage.

    Parameters
    ----------
    pose_dict : dict[str, np.ndarray]
        Maps session name -> pose array of shape (n_frames, n_keypoints, 3).
    window, stride : int
        Sliding window length / step.
    block_size : int
        Number of frames per contiguous block (e.g. 6000 = 1 minute at 100 fps).
    train_frac, val_frac, test_frac : float
        Fraction of each session's blocks assigned to each split. Must sum to 1.
    seed : int
        RNG seed for reproducible block assignment.
    save_dir : str or Path, optional
        If given, saves train_inds.npy / val_inds.npy / test_inds.npy and metadata.csv here.
    pose_h5_path : str or Path, optional
        If given, writes the full concatenated pose plus per-frame session ids here.

    Returns
    -------
    pose_full : np.ndarray
        Concatenated pose array (total_frames, n_keypoints, 3).
    train_inds, val_inds, test_inds : np.ndarray
        Window indices for each split (n_windows, window).
    metadata : pd.DataFrame
        Session metadata with path, name, session_offset, session_length.
        Index is session id (matches the ids array).
    """
    assert abs(train_frac + val_frac + test_frac - 1.0) < 1e-8

    session_names = list(pose_dict.keys())
    session_lengths = [pose_dict[s].shape[0] for s in session_names]
    session_offsets = np.concatenate([[0], np.cumsum(session_lengths)])[:-1]

    pose_full = np.concatenate([pose_dict[s] for s in session_names], axis=0)
    ids = np.concatenate([np.full(n, i) for i, n in enumerate(session_lengths)])

    if pose_h5_path is None and save_dir is not None:
        pose_h5_path = Path(save_dir) / "pose.h5"
    if pose_h5_path is not None:
        Path(pose_h5_path).parent.mkdir(parents=True, exist_ok=True)
        write.pose_h5(pose_full, ids, str(pose_h5_path))

    rng = np.random.default_rng(seed)

    train_inds, val_inds, test_inds = [], [], []
    for i, n in enumerate(session_lengths):
        if n < window:
            print(f"Skipping {session_names[i]}: length {n} < window {window}")
            continue

        start = session_offsets[i]
        frame_idx = np.arange(start, start + n)
        session_windows = sliding_window_view(frame_idx, window)[::stride]

        # Assign each window to a block by its start frame, then keep only windows whose
        # entire span (including a `gap`-frame margin on both ends) stays inside that
        # one block — this both prevents straddling a block boundary AND guarantees a
        # minimum gap between windows in blocks that end up in different splits.
        block_id = (session_windows[:, 0] - start) // block_size
        block_start = start + block_id * block_size
        block_end = np.minimum(block_start + block_size, start + n)

        margin_start = session_windows[:, 0] - block_start
        margin_end = block_end - session_windows[:, -1] - 1
        keep = (margin_end >= gap) # & (margin_start >= 0)

        session_windows = session_windows[keep]
        block_id = block_id[keep]

        blocks = rng.permutation(np.unique(block_id))

        n_train = int(np.floor(len(blocks) * train_frac))
        n_val = int(np.floor(len(blocks) * val_frac))

        train_blocks = blocks[:n_train]
        val_blocks = blocks[n_train : n_train + n_val]
        test_blocks = blocks[n_train + n_val :]

        train_inds.append(session_windows[np.isin(block_id, train_blocks)])
        val_inds.append(session_windows[np.isin(block_id, val_blocks)])
        test_inds.append(session_windows[np.isin(block_id, test_blocks)])

    train_inds = np.concatenate(train_inds, axis=0)
    val_inds = np.concatenate(val_inds, axis=0)
    test_inds = np.concatenate(test_inds, axis=0)

    train_inds = train_inds[np.argsort(train_inds[:, 0])]
    val_inds = val_inds[np.argsort(val_inds[:, 0])]
    test_inds = test_inds[np.argsort(test_inds[:, 0])]

    assert len(np.intersect1d(train_inds.flatten(), val_inds.flatten())) == 0
    assert len(np.intersect1d(train_inds.flatten(), test_inds.flatten())) == 0
    assert len(np.intersect1d(val_inds.flatten(), test_inds.flatten())) == 0

    # Create metadata DataFrame indexed by session id
    metadata = pd.DataFrame({
        'session_name': session_names,
        'session_offset': session_offsets,
        'session_length': session_lengths,
    })
    metadata.index.name = 'ids'

    if save_dir is not None:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        np.save(save_dir / "train_inds.npy", train_inds)
        np.save(save_dir / "val_inds.npy", val_inds)
        np.save(save_dir / "test_inds.npy", test_inds)
        metadata.to_csv(save_dir / "meta.csv")

    return pose_full, train_inds, val_inds, test_inds, metadata


def preprocess_virtual_rodent_data(
    data_path: str,
    skeleton_config: dict,
    window: int,
    train_val_test: str = "train",
    stride: int = 1,
    data_keys: List[str] = ["x6d", "root", "offsets"],
    direction_process: str = "midfwd",
    use_default_offsets: bool = False,
    **kwargs,
):
    """Load and preprocess virtual_rodent data.

    Loads concatenated pose from pose.h5, extracts windows using precomputed split indices,
    and computes x6d, offsets, root, and other features.

    Parameters
    ----------
    data_path : str
        Path to virtual_rodent directory
    skeleton_config : dict
        Skeleton configuration (KINEMATIC_TREE, OFFSET, LABELS)
    window : int
        Sliding window size
    train_val_test : str
        Which split ("train", "val", "test")
    stride : int
        Stride (informational, actual stride is in precomputed indices)
    data_keys : List[str]
        Features to compute (x6d, root, offsets, etc.)
    direction_process : str
        Direction processing mode ("midfwd" or "x360")
    use_default_offsets : bool
        Whether to use default offsets from constants

    Returns
    -------
    data : dict
        Dictionary with torch tensors for all requested data_keys
    """
    # Handle x3d which requires additional keys
    if "x3d" in data_keys:
        if "x6d" not in data_keys:
            data_keys = list(data_keys) + ["x6d"]
        if "offsets" not in data_keys:
            data_keys = list(data_keys) + ["offsets"]
        if "root" not in data_keys:
            data_keys = list(data_keys) + ["root"]

    data_path = Path(data_path)

    # Load full concatenated pose
    pose_full, ids = read.pose_h5(str(data_path / "pose.h5"))
    pose_full = pose_full.astype(np.float32)
    ids = ids.astype(np.int16)

    # Load precomputed split indices (n_windows, window_size)
    split_inds = np.load(data_path / f"{train_val_test}_inds.npy").astype(np.int32)

    # Extract windowed pose data: (n_windows, window, n_keypts, 3)
    pose = pose_full[split_inds]
    ids_split = ids[split_inds[:, window // 2]]  # IDs at center of each window

    data_len = len(pose)
    data = {}

    # Compute pose features (yaw, heading, x6d, offsets, root, target_pose)
    features = _preprocess_pose_to_features(
        pose,
        skeleton_config=skeleton_config,
        data_keys=data_keys,
        direction_process=direction_process,
        use_default_offsets=use_default_offsets,
        offsets_default=OFFSETS_3D.get("virtual_rodent"),
    )
    data.update(features)

    # Add IDs
    if "ids" in data_keys:
        data["ids"] = torch.tensor(ids_split, dtype=torch.int16)

    # Discrete classes for virtual_rodent (currently just one class)
    data["discrete_classes"] = {}

    # Validate all data has same length
    for k, v in data.items():
        if k == "discrete_classes":
            continue
        try:
            assert len(v) == data_len, f"Length mismatch for {k}: {len(v)} != {data_len}"
        except (AssertionError, TypeError):
            # offsets can have a different first dimension (n_keypoints,)
            if k == "offsets" and len(v.shape) == 2:
                assert v.shape[0] == len(skeleton_config["OFFSET"])
            else:
                raise

    return data

    return pose_full, train_inds, val_inds, test_inds, metadata