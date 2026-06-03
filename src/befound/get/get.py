import re
from pathlib import Path
import numpy as np
import befound
from befound.data import MouseDataset, MabeWindowDataset, fwd_kin_cont6d_torch, preprocess_save_data
from befound.data.constants import OFFSETS_3D, OFFSETS_3D_SUM
from torch.utils.data import DataLoader
import h5py
import pandas as pd
import torch
from torch import optim
from typing import List, Dict, Optional
from neuroposelib import read

import os
import copy

def fill_holes(data):
    clean_data = copy.deepcopy(data)
    
    for m in range(3):
        holes = np.where(clean_data[0, m, :, 0] == 0)[0]  # [0] 直接取出 1D array
        for h in holes:
            sub = np.where(clean_data[:, m, h, 0] != 0)[0]
            if sub.size > 0:
                clean_data[0, m, h, :] = clean_data[sub[0], m, h, :]
            else:
                return np.empty((0))

    for fr in range(1, clean_data.shape[0]):
        for m in range(3):
            holes = np.where(clean_data[fr, m, :, 0] == 0)[0]
            for h in holes:
                clean_data[fr, m, h, :] = clean_data[fr - 1, m, h, :]

    return clean_data

def get_mabe22_data(path, split="train", reindex=True):

    # def _load(filename):
    #     data = np.load(os.path.join(path, filename), allow_pickle=True).item()
    #     return np.stack([d["keypoints"] for d in data["sequences"].values()])
    def _load(filename):
        data = np.load(os.path.join(path, filename), allow_pickle=True).item()
        raw = np.stack([d["keypoints"] for d in data["sequences"].values()])
        # raw: (N, T, 3, K, 2)

        cleaned = []
        for i in range(raw.shape[0]):
            seq = raw[i]                   # (T, 3, K, 2)
            result = fill_holes(seq)
            if result.size == 0:
                print(f"[warn] sequence {i} has unfillable holes, skipping")
                continue
            cleaned.append(result)

        return np.stack(cleaned)           # (N', T, 3, K, 2)
    

    def _flatten_and_finalize(kp):
        # kp: (N, T, n_mice, K, 2) -> (N*n_mice, T, K, 2)  + per-mouse video idx
        N, T, n_mice, K, _ = kp.shape
        flat = kp.transpose(0, 2, 1, 3, 4).reshape(-1, T, K, 2)
        batch = np.repeat(np.arange(N), n_mice)
        if reindex:
            flat = flat[:, :, befound.data.constants.MABE_REINDEX, :]
        return flat.astype(np.float32, copy=False), batch

    if split == "all":
        kp_train = _load("mouse_triplet_train.npy")
        kp_sub   = _load("mouse_triplet_test.npy")
        N_train  = kp_train.shape[0]
        kp_full  = np.concatenate([kp_train, kp_sub], axis=0)
        flat, batch = _flatten_and_finalize(kp_full)
        split_mask = np.zeros(N_train + kp_sub.shape[0], dtype=bool)
        split_mask[:N_train] = True
        return flat, split_mask, batch

    if split == "train":
        kp = _load("mouse_triplet_train.npy")
    elif split == "submission":
        kp = _load("mouse_triplet_test.npy")
    else:
        raise ValueError(f"Invalid split: {split!r}")

    return _flatten_and_finalize(kp)

def get_mabe22_data_model(
    config: dict,
    train_val_test=["train","val"],
    # data_keys=["x2d", "offsets", "target_pose"],
    use_default_offsets=[True, True],
    shuffle=[True, False],
    # split: str = "train",
    stride: int = 6,
    num_workers: int = 2,
):
    """
    One-stop loader: returns a DataLoader yielding {"pose": (B, W, K, 2)}.
    """
    
    # val_data_keys = [
    #             "x2d",
    #             "offsets",
    #             "target_pose",
    #         ]
    mabe_path = config["data"]["data_path"]
    data_config = config["data"]
    epoch = config["model"]["start_epoch"]
    load_model = config["model"]["load_model"]
    window = config["model"]["window"]

    loader_dict = {}
    for is_shuffle, is_default_offsets, dataset_label in zip(shuffle, use_default_offsets, train_val_test):
        # curr_data_keys = val_data_keys if dataset_label == "val" else data_keys

        if dataset_label == "val":
            split = "submission"
            stride = 20
            pad_mode = "edge"
        else:            
            split = "train"
            pad_mode = "edge"

        kp, _batch = get_mabe22_data(mabe_path, split=split, reindex=True)
        ds = MabeWindowDataset(kp, window=window, stride=stride, pad_mode=pad_mode)
        loader_dict[dataset_label] = DataLoader(
            ds,
            batch_size=data_config["batch_size"],
            shuffle=is_shuffle,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=True,                # so train batches are uniform
            persistent_workers=num_workers > 0,
        )

    model = get_model(
        model_config=config["model"],
        load_model=load_model,
        epoch=epoch,
        n_keypts=loader_dict[train_val_test[0]].dataset.n_keypts,
        device="cuda",
        verbose=True,
    )

    return loader_dict, model

def data_and_model(
    config,
    load_model=None,
    epoch=None,
    train_val_test=["train", "val", "test"],
    data_keys=["x6d", "root", "offsets"],
    shuffle=[False, False, False],
    use_default_val_keys=True,
    use_default_offsets=[True, True, True],
    verbose=1,
):  
    if use_default_val_keys:
        if config["data"]["dataset"] == "4_mice":
            val_data_keys = [
                "ids",
                "x3d",
                "x6d",
                "root",
                "offsets",
                "target_pose",
            ]
        # if config["data"]["dataset"] == "mabe22":
        #     val_data_keys = [
        #         "x2d",
        #         "offsets",
        #         "target_pose",
        #     ]
        else:
            val_data_keys = [
                "ids",
                "x3d",
                "x6d",
                "root",
                "offsets",
                "target_pose",
            ]
    else:
        val_data_keys = data_keys

    if epoch is None:
        epoch = config["model"]["start_epoch"]

    if load_model is None:
        load_model = config["model"]["load_model"]

    ### Load Dataset
    # if train_val_test == "all":
    loader_dict = {}
    for is_shuffle, is_default_offsets, dataset_label in zip(shuffle, use_default_offsets, train_val_test):
        curr_data_keys = val_data_keys if dataset_label == "val" else data_keys
        loader_dict[dataset_label] = get_mouse_data(
            data_config=config["data"],
            train_val_test=dataset_label,
            data_keys=curr_data_keys,
            shuffle=is_shuffle,
            use_default_offsets=is_default_offsets,
        )

    model = get_model(
        model_config=config["model"],
        load_model=load_model,
        epoch=epoch,
        n_keypts=loader_dict[train_val_test[0]].dataset.n_keypts,
        device="cuda",
        verbose=verbose,
    )

    return loader_dict, model


def all_saved_epochs(path):
    z_path = Path(path + "weights/")
    epochs = [re.findall(r"\d+", f.parts[-1]) for f in list(z_path.glob("epoch*"))]
    epochs = np.sort(np.array(epochs).astype(int).squeeze())
    print("Epochs found: {}".format(epochs))

    return epochs


def get_mouse_data(
    data_config: dict,
    train_val_test: str = "train",
    data_keys: List[str] = ["x6d", "root", "offsets"],
    shuffle: bool = False,
    stride: Optional[int] = None,
    window: Optional[int] = None,
    use_default_offsets: bool = True,
):
    """
    Load in mouse data and return pytorch dataloaders
    """
    skeleton_config = read.config(
        "{}mouse_skeleton.yaml".format(data_config["data_path"])
    )

    if train_val_test != "full":
        if data_config["dataset"] == "parkinsons_healthy":
            dataset_name = "parkinsons"
        else:
            dataset_name = data_config["dataset"]
        data_path = "{}{}/{}/".format(
            data_config["data_path"], dataset_name, train_val_test
        )
        if "x3d" in data_keys:
            if "x6d" not in data_keys:
                data_keys += ["x6d"]

            if "offsets" not in data_keys:
                data_keys += ["offsets"]

            if "root" not in data_keys:
                data_keys += ["root"]

        data = {}
        for key in data_keys + ["ids"]:
            if key in ["pd_label", "fluorescence", "x3d"]:
                continue
            elif key in ["ids", "heading", "avg_speed_3d", "raw_pose"]:
                file_path = "{}{}.h5".format(data_path, key)
            elif key == "offsets":
                if use_default_offsets:
                    data["offsets"] = OFFSETS_3D
                    print("OFFSETS SUM: {:.3f}".format(OFFSETS_3D_SUM))
                    data["offsets"] = data["offsets"][:, None] * np.array(
                        skeleton_config["OFFSET"], dtype=np.float32
                    )
                    continue
                else:
                    file_path = "{}{}.h5".format(data_path, key)
            else:
                file_path = "{}{}_{}.h5".format(
                    data_path, key, data_config["direction_process"]
                )
            print("Reading in {} from {}".format(key, file_path))
            hf = h5py.File(file_path, "r")
            data[key] = np.array(hf.get(key))
            hf.close()

        data = {k: torch.from_numpy(v) for k, v in data.items()}

        # if "x3d" in data_keys:
        #     reshaped_x6d = data["x6d"].reshape((-1,) + data["x6d"].shape[-2:])
        #     offsets = data["offsets"]
        #     root = data["root"].reshape((-1, 3)) / offsets_sum
        #     # data["x3d"] = fwd_kin_cont6d_torch(
        #     #     reshaped_x6d,
        #     #     skeleton_config["KINEMATIC_TREE"],
        #     #     offsets,
        #     #     root_pos=root,
        #     #     do_root_R=True,
        #     #     eps=1e-8,
        #     # ).reshape(data["x6d"].shape[:-1] + (3,))

        #     if "target_pose" in data_keys:
        #         data["target_pose"] /= offsets_sum

    elif train_val_test == "full":
        data = preprocess_save_data(
            data_path=data_config["data_path"],
            skeleton_config=skeleton_config,
            dataset=data_config["dataset"],
            window=window,
            stride=stride,
            data_keys=data_keys + ["ids"],
            speed_threshold=2.25,
            use_default_offsets=use_default_offsets,
            direction_process=data_config["direction_process"],
        )

    discrete_classes = {}
    if data_config["dataset"] == "parkinsons":
        # Only if read in raw poses for the PD dataset
        # if not ((data_config["stride"] == 5) or (data_config["stride"] == 10)):
        if "pd_label" in data_keys:
            data["pd_label"] = torch.zeros((len(data["ids"]), 1)).long()
            data["pd_label"][data["ids"] >= 36] = 1
            discrete_classes["pd_label"] = torch.unique(data["pd_label"], sorted=True)

        if "fluorescence" in data_keys:
            meta = pd.read_csv(
                data_config["data_path"] + data_config["dataset"] + "/metadata.csv"
            )
            # import pdb; pdb.set_trace()
            meta_by_frame = meta.iloc[data["ids"]]
            fluorescence = meta_by_frame["Fluorescence"].to_numpy()
            data["fluorescence"] = torch.tensor(fluorescence, dtype=torch.float32)

        data["ids"][data["ids"] >= 36] = data["ids"][data["ids"] >= 36] - 36
        unique_ids = torch.unique(data["ids"])
        discrete_classes["ids"] = torch.arange(len(unique_ids)).long()
    else:
        discrete_classes["ids"] = torch.unique(data["ids"], sorted=True)

    dataset = MouseDataset(
        data,
        data_config["arena_size"],
        skeleton_config["KINEMATIC_TREE"],
        len(skeleton_config["LABELS"]),
        label=train_val_test,
        discrete_classes=discrete_classes,
        offsets_sum=OFFSETS_3D_SUM,
    )
    loader = DataLoader(
        dataset=dataset,
        batch_size=data_config["batch_size"],
        shuffle=shuffle,
        num_workers=5,
        pin_memory=True,
    )

    return loader


def get_model(
    model_config,
    load_model,
    epoch,
    n_keypts: int=51,
    device:str="cuda",
    verbose=1,
):

    ### Initialize/load model
    if (model_config["type"] == "cnn") and (model_config["is_2d"]):
        from befound.model.vae import ResVAE2D

        in_channels = n_keypts * 2

        vae = ResVAE2D(
            in_channels=in_channels,
            out_channels=n_keypts * 2,
            hidden_dim=model_config["hidden_dim"],
            latent_dim=model_config["latent_dim"],
            depth=model_config["depth"],
            activation=model_config["activation"],
            n_keypts=n_keypts,
            out_kernel_size=model_config["out_kernel_size"],
            prior=model_config["prior"],
            window_size=model_config["window"],
        )
    elif (model_config["type"] == "ci_cnn") and (model_config["is_2d"]):
        from befound.model.vae import CIResVAE2D

        in_channels = 2
        vae = CIResVAE2D(
            in_channels=in_channels,
            out_channels=n_keypts * 2,
            hidden_dim=model_config["hidden_dim"],
            latent_dim=model_config["latent_dim"],
            depth=model_config["depth"],
            query_size=model_config["query_size"],
            activation=model_config["activation"],
            n_keypts=n_keypts,
            out_kernel_size=model_config["out_kernel_size"],
            prior=model_config["prior"],
            window_size=model_config["window"],
        )
    elif (model_config["type"] == "cnn") and not (model_config["is_2d"]):
        from befound.model.vae import ResVAE

        in_channels = n_keypts * 3

        vae = ResVAE(
            in_channels=in_channels,
            out_channels=n_keypts * 6 + 3,
            hidden_dim=model_config["hidden_dim"],
            latent_dim=model_config["latent_dim"],
            depth=model_config["depth"],
            activation=model_config["activation"],
            n_keypts=n_keypts,
            out_kernel_size=model_config["out_kernel_size"],
            prior=model_config["prior"],
            window_size=model_config["window"],
        )

    elif (model_config["type"] == "ci_cnn") and not (model_config["is_2d"]):
        from befound.model.vae import CIResVAE

        in_channels = 3
        vae = CIResVAE(
            in_channels=in_channels,
            out_channels=n_keypts * 6 + 3,
            hidden_dim=model_config["hidden_dim"],
            latent_dim=model_config["latent_dim"],
            depth=model_config["depth"],
            query_size=model_config["query_size"],
            activation=model_config["activation"],
            n_keypts=n_keypts,
            out_kernel_size=model_config["out_kernel_size"],
            prior=model_config["prior"],
            window_size=model_config["window"],
        )

    if verbose > 0:
        print(vae)

    if load_model is not None:
        load_path = "{}/weights/epoch_{}.pth".format(load_model, epoch)
        print("Loading Weights from:\n{}".format(load_path))
        state_dict = torch.load(load_path)
        missing_keys, unexpected_keys = vae.load_state_dict(state_dict, strict=False)

        if verbose > 0:
            print("Missing Keys: {}".format(missing_keys))
            print("Unexpected Keys: {}".format(unexpected_keys))

    return vae.to(device)


