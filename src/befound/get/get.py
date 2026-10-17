import re
from pathlib import Path
import numpy as np
from befound.data import RodentDiskDataset
from befound.data.constants import OFFSETS_3D  # , BODY_SCALE
from torch.utils.data import DataLoader
import h5py
import pandas as pd
import torch
from typing import List, Dict, Optional, Union
from neuroposelib import read


def data_and_model(
    config,
    load_model=None,
    epoch=None,
    train_val_test=["train", "val", "test"],
    shuffle=[False, False, False],
    verbose=1,
    balanced_sampling: bool = True,
    use_all_test_val_data: bool = False,
):

    # if use_default_val_keys:
    #     val_data_keys = [
    #         "ids",
    #         "x3d",
    #         "x6d",
    #         "root",
    #         "offsets",
    #         "target_pose",
    #     ]
    # else:
    #     val_data_keys = data_keys

    if epoch is None:
        epoch = config["model"]["start_epoch"]

    if load_model is None:
        load_model = config["model"]["load_model"]

    ### Load Dataset
    # if train_val_test == "all":
    loader_dict = {}
    for is_shuffle, split in zip(shuffle, train_val_test):
        data_config = config["data"].copy()
        if split in ["val", "test"] and use_all_test_val_data:
            # dataset_id is assigned by position in this list (see
            # RodentDiskDataset), and the model's per-dataset decoder/encoder
            # heads are built from config["data"]["datasets"]'s order (see
            # get_model below) -- so any dataset also used in training must
            # keep its original position here, with eval-only datasets
            # (missing from training) appended after, not spliced in.
            all_datasets = [
                "wu_iclr25",
                "pairr24m",
                "mouse_ephys",
                "sdannce",
                "virtual_rodent",
            ]
            data_config["datasets"] = config["data"]["datasets"] + [
                name for name in all_datasets if name not in config["data"]["datasets"]
            ]

        # curr_data_keys = val_data_keys if split == "val" else data_keys
        # if from_h5:
        loader_dict[split] = get_rodent_data_disk(
            data_config=data_config,
            train_val_test=split,
            shuffle=is_shuffle,
            # use_default_offsets=is_default_offsets,
            use_pe_indices=config["model"]["use_be"],
            model_window=config["model"]["window"],
            min_interp_fps=config["train"]["augmentations"]["fps"]["min_fps"],
            balanced_sampling=balanced_sampling,
        )
        # else:
        #     loader_dict[split] = get_rodent_data(
        #         data_config=config["data"],
        #         train_val_test=split,
        #         data_keys=curr_data_keys,
        #         shuffle=is_shuffle,
        #         use_default_offsets=is_default_offsets,
        #         demo=demo,
        #     )

    discrete_classes = {}
    # if "is_2d" in config["disentangle"].keys():
    #     discrete_classes["is_2d"] = [0, 1]
    if "2d_project" in config["disentangle"].keys():
        discrete_classes["2d_project"] = [0, 1]
    if "dataset_id" in config["disentangle"].keys():
        # Deliberately scoped to the training datasets only -- eval-only
        # datasets appended above (see use_all_test_val_data) exist purely
        # for per-dataset jpe/root metrics and must stay out of the
        # disentangling bottleneck, not get their own learned class. Their
        # dataset_id values fall outside this range; decode() handles that
        # (see ResVAEBase.decode's one-hot construction).
        discrete_classes["dataset_id"] = [
            i for i in range(len(config["data"]["datasets"]))
        ]
    # The encoder has to process every sample in a batch, including any
    # eval-only datasets appended above -- so unlike out_channels/n_keypts
    # (kept training-only, below), its n_keypts/dataset_names need to cover
    # the union of every loader actually built here. model_dataset_names
    # is whichever loader's own list is longest, since that list's order
    # already matches how its RodentDiskDataset assigned dataset_id (training
    # datasets first, in their original order, eval-only ones appended after
    # -- see use_all_test_val_data above), and merging n_keypts dicts is
    # order-independent.
    model_dataset_names = max(
        (l.dataset.dataset_names for l in loader_dict.values()), key=len
    )
    model_n_keypts = {}
    for l in loader_dict.values():
        model_n_keypts.update(l.dataset.n_keypts)

    model = get_model(
        model_config=config["model"],
        load_model=load_model,
        epoch=epoch,
        n_keypts=loader_dict[train_val_test[0]].dataset.n_keypts,
        dataset_names=config["data"]["datasets"],
        model_n_keypts=model_n_keypts,
        model_dataset_names=model_dataset_names,
        disentangle_config=config["disentangle"],
        loss_config=config["loss"],
        discrete_classes=discrete_classes,
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
    Custom collate function for multi-dataset batches with variable keypoint counts.

    Pads all tensors to max n_keypts (44), then indexes to actual keypoints before
    encoder forward pass. This approach provides ragged efficiency while enabling
    multiprocessing and simpler tensor handling.

    Parameters
    ----------
    batch : List[Dict]
        List of samples from RodentDataset with variable keypoint counts

    Returns
    -------
    Dict
        Collated batch with padded tensors. Includes dataset_idx_to_n_keypts for indexing.
    """
    collated = {}

    if not batch:
        return collated

    # Get all keys from first sample
    sample_keys = batch[0].keys()

    # Find max n_keypts for padding keypoint-dimension tensors
    max_n_keypts = 0
    for sample in batch:
        if "x6d" in sample:
            n_keypts = sample["x6d"].shape[1]
            max_n_keypts = max(max_n_keypts, n_keypts)

    for key in sample_keys:
        values = [sample[key] for sample in batch]

        if key in ["dataset_id"]:
            # Stack scalars normally
            collated[key] = torch.stack(values)
        elif key == "dataset_name":
            # Keep dataset names as list for reference
            collated[key] = values
        elif isinstance(values[0], torch.Tensor):
            # Pad keypoint-dimension tensors to max_n_keypts
            if key in ["x6d", "offsets", "x3d", "target_pose"]:
                padded_values = []
                for v in values:
                    if v.shape[1] < max_n_keypts:
                        # Pad dimension 1 (keypoint axis) with zeros
                        pad_amount = max_n_keypts - v.shape[1]
                        v_padded = torch.nn.functional.pad(
                            v, (0, 0, 0, 0, 0, pad_amount)
                        )
                        padded_values.append(v_padded)
                    else:
                        padded_values.append(v)
                collated[key] = torch.stack(padded_values)
            else:
                # Stack normally (these don't vary by keypoint count)
                collated[key] = torch.stack(values)
        elif isinstance(values[0], dict):
            # discrete_classes or other dict - keep as is
            collated[key] = values[0]
        else:
            # Other types (lists, scalars, etc.)
            collated[key] = values

    return collated


def collate_h5_dataset(batch: List[Dict]) -> Dict:
    """
    Custom collate function for RodentDiskDataset batches.

    Samples are already zero-padded to a common n_keypts inside
    RodentDiskDataset.__getitem__, so no padding happens here. This only
    exists to stack via torch.stack instead of the default collate_fn's
    num_workers>0 fast path, which pre-allocates shared-memory storage and
    resizes into it -- a step that fails on tensors created via
    torch.from_numpy (their storage is marked non-resizable).
    """
    collated = {}
    for key in batch[0].keys():
        values = [sample[key] for sample in batch]
        collated[key] = (
            torch.stack(values) if isinstance(values[0], torch.Tensor) else values
        )
    return collated


def get_rodent_data_disk(
    data_config: dict,
    train_val_test: str = "train",
    shuffle: bool = False,
    use_default_seg_len: bool = True,
    use_pe_indices: bool = True,
    model_window: int = 51,
    min_interp_fps: Optional[int] = None,
    balanced_sampling: bool = True,
):
    skeleton_configs, kinematic_trees, skeleton_offsets = {}, {}, {}
    # pe_indices = {} if use_be else None
    # for i, dataset_name in enumerate(data_config["datasets"]):
    #     skeleton_configs[dataset_name] = read.config(
    #         f"{data_config['data_path']}/{dataset_name}/skeleton.yaml"
    #     )

    #     kinematic_trees[dataset_name] = skeleton_configs[dataset_name]["KINEMATIC_TREE"]

    #     skeleton_offsets[dataset_name] = torch.tensor(
    #         skeleton_configs[dataset_name]["OFFSET"], dtype=torch.float32
    #     )

    #     if use_be:
    #         pe_indices[dataset_name] = torch.tensor(skeleton_configs[dataset_name]["PE_INDICES"], dtype=torch.long)

    # strides = {k: v for k, v in zip(data_config["datasets"], data_config["stride"])}
    dataset = RodentDiskDataset(
        data_path=data_config["data_path"],
        dataset_names=data_config["datasets"],
        train_val_test=train_val_test,
        # kinematic_tree=kinematic_trees,
        # pe_indices=pe_indices,
        stride=data_config["stride"],
        use_pe_indices=use_pe_indices,
        # skeleton_offsets=skeleton_offsets,
        use_default_seg_len=use_default_seg_len,
        model_window=model_window,
        min_interp_fps=min_interp_fps,
        balanced_sampling=balanced_sampling,
        virtual_rodent_x_splits=data_config.get("virtual_rodent_x_splits", []),
        use_filtered_inds=data_config["use_filtered_inds"],
        scale_body=True,
    )  # train_val_test=train_val_test, data_keys=data_keys, use_default_offsets=use_default_offsets)

    sampler = dataset.get_sampler(shuffle)

    loader = DataLoader(
        dataset=dataset,
        batch_size=data_config["batch_size"],
        shuffle=shuffle if sampler is None else False,
        sampler=sampler,
        num_workers=8,  # Padded tensors serialize well; parallel data loading while GPU trains
        pin_memory=True,
        prefetch_factor=2,
        collate_fn=collate_h5_dataset,
    )

    return loader


# def get_rodent_data(
#     data_config: dict,
#     train_val_test: str = "train",
#     data_keys: List[str] = ["x6d", "root", "offsets"],
#     shuffle: bool = False,
#     use_default_offsets: bool = True,
#     demo: bool = False,
# ):
#     """
#     Load in mouse data from one or multiple datasets and return pytorch dataloader.

#     Supports variable numbers of keypoints across datasets via custom collate function.
#     x6d tensors are kept as ragged arrays (list) to accommodate different keypoint counts.
#     """
#     from befound.data.datasets import (
#         preprocess_wu_iclr25_data,
#         preprocess_pairr24m_data,
#         preprocess_mouse44_ephys_data,
#         preprocess_virtual_rodent_data,
#         preprocess_sdannce_data,
#     )

#     # Handle both old single-dataset and new multi-dataset config formats
#     if "datasets" in data_config:
#         dataset_names = data_config["datasets"]
#     else:
#         raise ValueError("Config must have 'datasets' key")

#     # Map dataset names to their preprocessing functions
#     preprocess_funcs = {
#         "wu_iclr25": preprocess_wu_iclr25_data,
#         "pairr24m": preprocess_pairr24m_data,
#         "mouse44_ephys": preprocess_mouse44_ephys_data,
#         "virtual_rodent": preprocess_virtual_rodent_data,
#         "sdannce": preprocess_sdannce_data,
#     }

#     # Load data from each dataset into a dict
#     datasets_dict = {}
#     skeleton_configs = {}

#     for i, dataset_name in enumerate(dataset_names):
#         if (train_val_test == "full") and (dataset_name == "wu_iclr25"):
#             print(
#                 "`full` split is not supported for dataset: {}. Using `train` split instead.".format(
#                     dataset_name
#                 )
#             )
#             temp_train_val_test = "train"
#         else:
#             temp_train_val_test = train_val_test
#         print(f"\nLoading dataset: {dataset_name}")

#         # Get the appropriate preprocessing function
#         if dataset_name == "wu_iclr25_healthy":
#             func_key = "wu_iclr25"
#             actual_dataset_name = "wu_iclr25"
#         else:
#             func_key = dataset_name
#             actual_dataset_name = dataset_name

#         if func_key not in preprocess_funcs:
#             raise ValueError(
#                 f"Unknown dataset: {dataset_name}. Available: {list(preprocess_funcs.keys())}"
#             )

#         preprocess_func = preprocess_funcs[func_key]
#         skeleton_configs[actual_dataset_name] = read.config(
#             f"{data_config['data_path']}/{actual_dataset_name}/skeleton.yaml"
#         )

#         # Load dataset using appropriate preprocessing function
#         dataset_data = preprocess_func(
#             data_path=f"{data_config['data_path']}/{actual_dataset_name}/",
#             skeleton_config=skeleton_configs[actual_dataset_name],
#             window=data_config.get("window", 51),
#             train_val_test=temp_train_val_test,
#             stride=data_config.get("stride", 10)[i],
#             data_keys=data_keys + ["ids"],
#             direction_process=data_config.get("direction_process", "midfwd"),
#             use_default_offsets=use_default_offsets,
#             demo=demo,
#         )

#         # Store dataset with its original dataset-specific IDs (no remapping)
#         datasets_dict[actual_dataset_name] = dataset_data

#     # Create per-dataset metadata dicts
#     kinematic_tree_dict = {
#         name: skeleton_configs[name]["KINEMATIC_TREE"] for name in dataset_names
#     }
#     n_keypts_dict = {
#         name: len(skeleton_configs[name]["LABELS"]) for name in dataset_names
#     }
#     body_scale_dict = {name: BODY_SCALE[name] for name in dataset_names}

#     # Create RodentDataset in multi-dataset mode
#     dataset = RodentDataset(
#         data=datasets_dict,
#         kinematic_tree=kinematic_tree_dict,
#         n_keypts=n_keypts_dict,
#         body_scale=body_scale_dict,
#         include_dataset_id=True,
#     )
#     loader = DataLoader(
#         dataset=dataset,
#         batch_size=data_config["batch_size"],
#         shuffle=shuffle,
#         num_workers=4,  # Padded tensors serialize well; parallel data loading while GPU trains
#         pin_memory=True,
#         collate_fn=collate_variable_keypoints,  # Pads to max n_keypts, indexes before encoder
#     )

#     return loader


def get_model(
    model_config,
    load_model,
    epoch,
    n_keypts: Union[int, Dict[str, int]] = 51,
    dataset_names: List[str] = [],
    model_n_keypts: Optional[Dict[str, int]] = None,
    model_dataset_names: Optional[List[str]] = None,
    disentangle_config: Optional[Dict] = {},
    loss_config: Optional[Dict] = {},
    discrete_classes: Optional[Dict] = {},
    device: str = "cuda",
    verbose: int = 1,
):
    """
    Initialize and optionally load a VAE model.

    Parameters
    ----------
    n_keypts : int or Dict[str, int]
        For single-dataset: scalar keypoint count
    model_n_keypts, model_dataset_names : Dict[str, int], List[str], optional
        Used for the encoder's own per-dataset keypoint lookup instead of
        n_keypts/dataset_names, if given -- the encoder has to process every
        sample in a batch, including eval-only datasets that aren't among
        n_keypts/dataset_names (which stay training-only, since they also
        determine the decoder's out_channels/learned head count and must
        match a loaded checkpoint). Defaults to n_keypts/dataset_names.
        For multi-dataset: dict mapping dataset_name -> keypoint count
    """

    feat_dim_dict = {"rotation": 6, "fps": 1}

    # # Conditional VAE adding dimensions to decoder input
    # if "conditional" in methods.keys():  # len(methods["conditional"]) > 0:
    #     conditional_dim = sum([feat_dim_dict[k] for k in methods["conditional"]])
    #     conditional_keys = methods["conditional"]
    # else:
    #     conditional_keys = None
    #     conditional_dim = 0

    model_n_keypts = model_n_keypts if model_n_keypts is not None else n_keypts
    model_dataset_names = (
        model_dataset_names if model_dataset_names is not None else dataset_names
    )

    scrubber_dict = {}
    scrub_features = list(disentangle_config.keys())
    # discrete_classes (built in data_and_model from config["disentangle"]) is the
    # source of truth for categorical scrub_features ("is_2d", "dataset_id", ...);
    # skip them here so their dims aren't counted twice.
    conditional_dim = sum(
        [feat_dim_dict[k] for k in scrub_features if k not in discrete_classes]
    )
    conditional_dim += sum([len(v) for v in discrete_classes.values()])
    for feature in disentangle_config.keys():
        method = disentangle_config[feature]["method"]
        # Moving Average Least Squares Filter with n-order polynomical features
        if method == "moving_avg_lsq":  # len(methods["moving_avg_lsq"]) > 0:
            from scrubvae.model.disentangle import MovingAvgLeastSquares

            if "moving_avg_lsq" not in scrubber_dict.keys():
                scrubber_dict["moving_avg_lsq"] = {}
            # for feat in methods["moving_avg_lsq"]:
            scrubber_dict["moving_avg_lsq"][feature] = MovingAvgLeastSquares(
                model_config["latent_dim"],
                feat_dim_dict[feature],
                bias=loss_config[feature + "_mals"] < 0,
                polynomial_order=disentangle_config[feature]["polynomial"],
                l2_reg=disentangle_config[feature]["l2_reg"],
            )

        # Quadratic Discriminant Filter for class scrubbing
        if method == "qda":  # len(methods["qda"]) > 0:
            from scrubvae.model.disentangle import QuadraticDiscriminantFilter

            if "qda" not in scrubber_dict.keys():
                scrubber_dict["qda"] = {}
            scrubber_dict["qda"][feature] = QuadraticDiscriminantFilter(
                model_config["latent_dim"], discrete_classes[feature]
            )

        # Moving Average Filter for class scrubbing
        if method == "moving_avg":  # len(methods["moving_avg"]) > 0:
            from scrubvae.model.disentangle import MovingAverageFilter

            if "moving_avg" not in scrubber_dict.keys():
                scrubber_dict["moving_avg"] = {}
            # for feat in methods["moving_avg"]:
            scrubber_dict["moving_avg"][feature] = MovingAverageFilter(
                model_config["latent_dim"], discrete_classes[feature]
            )

    # Helper function to compute channel dimensions
    if model_config["is_2d"]:
        in_channels = {k: v * 2 for k, v in n_keypts.items()}
        out_channels = in_channels
    else:
        in_channels = {k: v * 2 for k, v in n_keypts.items()}
        out_channels = {k: v * 6 + 3 for k, v in n_keypts.items()}

    # dataset_names = list(n_keypts.keys())
    if model_config["type"] == "ci_cnn":
        in_channels = 2 if model_config["is_2d"] else 3

    ### Initialize/load model
    if model_config["type"] == "cnn":
        from befound.model.vae import ResVAE

        vae = ResVAE(
            in_channels=in_channels,
            out_channels=out_channels,
            hidden_dim=model_config["hidden_dim"],
            latent_dim=model_config["latent_dim"],
            depth=model_config["depth"],
            activation=model_config["activation"],
            n_keypts=model_n_keypts,
            dataset_names=model_dataset_names,
            out_kernel_size=model_config["out_kernel_size"],
            prior=model_config["prior"],
            window_size=model_config["window"],
            is_2d=model_config["is_2d"],
            scrubber_dict=scrubber_dict,
            conditional_dim=conditional_dim,
            discrete_classes=discrete_classes,
        )
    elif model_config["type"] == "ci_cnn":
        from befound.model.vae import CIResVAE

        vae = CIResVAE(
            in_channels=in_channels,
            out_channels=out_channels,
            hidden_dim=model_config["hidden_dim"],
            latent_dim=model_config["latent_dim"],
            depth=model_config["depth"],
            query_size=model_config["query_size"],
            activation=model_config["activation"],
            n_keypts=model_n_keypts,
            dataset_names=model_dataset_names,
            out_kernel_size=model_config["out_kernel_size"],
            prior=model_config["prior"],
            window_size=model_config["window"],
            is_2d=model_config["is_2d"],
            use_be=model_config["use_be"],
            scrubber_dict=scrubber_dict,
            conditional_dim=conditional_dim,
            discrete_classes=discrete_classes,
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
