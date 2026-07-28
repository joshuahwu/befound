import re
from pathlib import Path
import numpy as np
from befound.data import RodentDataset
from befound.data.constants import OFFSETS_3D, OFFSETS_3D_SUM
from torch.utils.data import DataLoader
import h5py
import pandas as pd
import torch
from typing import List, Dict, Optional
from neuroposelib import read


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
    for is_shuffle, is_default_offsets, split in zip(shuffle, use_default_offsets, train_val_test):
        curr_data_keys = val_data_keys if split == "val" else data_keys
        loader_dict[split] = get_rodent_data(
            data_config=config["data"],
            train_val_test=split,
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


def collate_variable_keypoints(batch: List[Dict]) -> Dict:
    """
    Custom collate function for batches with variable numbers of keypoints.

    Handles cases where tensors have different shapes due to different keypoint counts
    across datasets (e.g., x6d, offsets, x3d, target_pose all vary by n_keypts).
    - Scalar metadata (dataset_id, dataset_name, ids) are stacked normally
    - Ragged tensors (different shapes per sample) are kept as lists for set transformer
    - Other tensors are stacked if possible, kept as lists if shapes differ

    Parameters
    ----------
    batch : List[Dict]
        List of samples from RodentDataset with variable keypoint counts

    Returns
    -------
    Dict
        Collated batch with stacked tensors and ragged lists for variable-shape data
    """
    collated = {}

    if not batch:
        return collated

    # Get all keys from first sample
    sample_keys = batch[0].keys()

    for key in sample_keys:
        values = [sample[key] for sample in batch]

        if key in ["dataset_id"]:
            # Stack scalars normally
            collated[key] = torch.stack(values)
        elif key == "dataset_name":
            # Keep dataset names as list for reference
            collated[key] = values
        elif isinstance(values[0], torch.Tensor):
            # Try to stack; if shapes differ (ragged tensors from variable keypoints), keep as list
            try:
                collated[key] = torch.stack(values)
            except RuntimeError:
                # Shapes don't match - keep as ragged list for set transformer
                # This handles x6d, offsets, x3d, target_pose, etc. that vary by n_keypts
                collated[key] = values
        elif isinstance(values[0], dict):
            # discrete_classes or other dict - keep as is
            collated[key] = values[0]
        else:
            # Other types (lists, scalars, etc.)
            collated[key] = values

    return collated


def get_rodent_data(
    data_config: dict,
    train_val_test: str = "train",
    data_keys: List[str] = ["x6d", "root", "offsets"],
    shuffle: bool = False,
    use_default_offsets: bool = True,
):
    """
    Load in mouse data from one or multiple datasets and return pytorch dataloader.

    Supports variable numbers of keypoints across datasets via custom collate function.
    x6d tensors are kept as ragged arrays (list) to accommodate different keypoint counts.
    """
    from befound.data.datasets import (
        preprocess_wu_iclr25_data,
        preprocess_pairr24m_data,
        preprocess_mouse44_ephys_data,
        preprocess_virtual_rodent_data,
    )

    # Handle both old single-dataset and new multi-dataset config formats
    if "datasets" in data_config:
        dataset_names = data_config["datasets"]
    else:
        raise ValueError("Config must have 'datasets' key")

    # Map dataset names to their preprocessing functions
    preprocess_funcs = {
        "wu_iclr25": preprocess_wu_iclr25_data,
        "wu_iclr25_healthy": preprocess_wu_iclr25_data,
        "pairr24m": preprocess_pairr24m_data,
        "mouse44_ephys": preprocess_mouse44_ephys_data,
        "virtual_rodent": preprocess_virtual_rodent_data,
    }

    # Load data from each dataset into a dict
    datasets_dict = {}
    skeleton_configs = {}

    for i, dataset_name in enumerate(dataset_names):
        print(f"\nLoading dataset: {dataset_name}")

        # Get the appropriate preprocessing function
        if dataset_name == "wu_iclr25_healthy":
            func_key = "wu_iclr25"
            actual_dataset_name = "wu_iclr25"
        else:
            func_key = dataset_name
            actual_dataset_name = dataset_name

        if func_key not in preprocess_funcs:
            raise ValueError(f"Unknown dataset: {dataset_name}. Available: {list(preprocess_funcs.keys())}")

        preprocess_func = preprocess_funcs[func_key]

        print("Calculating dataset: {}".format(actual_dataset_name))
        skeleton_configs[actual_dataset_name] = read.config(
            f"{data_config['data_path']}/{actual_dataset_name}/skeleton.yaml"
        )

        # Load dataset using appropriate preprocessing function
        dataset_data = preprocess_func(
            data_path=f"{data_config['data_path']}/{actual_dataset_name}/",
            skeleton_config=skeleton_configs[actual_dataset_name],
            window=data_config.get("window", 51),
            train_val_test=train_val_test,
            stride=data_config.get("stride", 10)[i],
            data_keys=data_keys + ["ids"],
            direction_process=data_config.get("direction_process", "midfwd"),
            use_default_offsets=use_default_offsets,
        )

        # Store dataset with its original dataset-specific IDs (no remapping)
        datasets_dict[actual_dataset_name] = dataset_data

    # Create per-dataset metadata dicts
    kinematic_tree_dict = {
        name: skeleton_configs[name]["KINEMATIC_TREE"]
        for name in dataset_names
    }
    n_keypts_dict = {
        name: len(skeleton_configs[name]["LABELS"])
        for name in dataset_names
    }
    offsets_sum_dict = {
        name: OFFSETS_3D_SUM[name]
        for name in dataset_names
    }

    # Create RodentDataset in multi-dataset mode
    dataset = RodentDataset(
        data=datasets_dict,
        kinematic_tree=kinematic_tree_dict,
        n_keypts=n_keypts_dict,
        offsets_sum=offsets_sum_dict,
        include_dataset_id=True,
    )
    loader = DataLoader(
        dataset=dataset,
        batch_size=data_config["batch_size"],
        shuffle=shuffle,
        num_workers=0,  # Ragged tensors (list) don't serialize well across multiprocessing; CPU loads sequentially while GPU trains
        pin_memory=True,
        collate_fn=collate_variable_keypoints,  # Custom collator for variable keypoint counts
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
