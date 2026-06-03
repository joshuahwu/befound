import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from befound.train.losses import get_batch_loss
import torch.optim as optim
import tqdm
import time
import wandb
from line_profiler import profile
from pathlib import Path
from befound.data.train_utils import get_x3d_from_data, prepare_batch, prepare_batch_2d_bespoke
import copy


class CyclicalBetaAnnealing(torch.nn.Module):
    def __init__(self, beta_max=1, len_cycle=100, R=0.5):
        self.beta_max = beta_max
        self.len_cycle = len_cycle
        self.R = R
        self.len_increasing = int(len_cycle * R)

    def get(self, epoch):
        remainder = (epoch - 1) % self.len_cycle
        if remainder >= self.len_increasing:
            beta = self.beta_max
        else:
            beta = self.beta_max * remainder / self.len_increasing

        return beta


def get_beta_schedule(schedule, beta):
    if schedule == "cyclical":
        print("Initializing cyclical beta annealing")
        beta_scheduler = CyclicalBetaAnnealing(beta_max=beta)
    else:
        print("No beta annealing selected")
        beta_scheduler = None

    return beta_scheduler


def get_optimizer_and_lr_scheduler(
    model, train_config, load_path=None, start_epoch=None
):
    """
    Loads in optimizer and learning rate schedulers
    """
    if train_config["optimizer"] == "adam":
        print("Initializing Adam optimizer ...")
        optimizer = optim.Adam(model.parameters(), lr=train_config["lr"])
    elif train_config["optimizer"] == "adamw":
        print("Initializing AdamW optimizer ...")
        optimizer = optim.AdamW(model.parameters(), lr=train_config["lr"])
    elif train_config["optimizer"] == "sgd":
        print("Initializing SGD optimizer ...")
        optimizer = optim.SGD(
            model.parameters(), lr=train_config["lr"], momentum=0.2, nesterov=True
        )
    else:
        raise ValueError("No valid optimizer selected")

    if train_config["lr_schedule"] == "cawr":
        print("Initializing cosine annealing w/warm restarts learning rate scheduler")
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=50)
    elif train_config["lr_schedule"] is None:
        print("No learning rate scheduler selected")
        scheduler = None

    if load_path is not None:
        if Path("{}/checkpoints/epoch_{}.pth".format(load_path, start_epoch)).exists():
            checkpoint = torch.load(
                "{}/checkpoints/epoch_{}.pth".format(load_path, start_epoch)
            )
            optimizer.load_state_dict(checkpoint["optimizer"])
            scheduler = checkpoint["lr_scheduler"]

    return optimizer, scheduler


def predict_batch(model, data):
    data_i = {k: v for k, v in data.items() if k in ["x3d", "x2d"]}

    return model(data_i)

@profile
def train_test_epoch(
    config,
    model,
    loader,
    device,
    epoch,
    optimizer=None,
    scheduler=None,
    mode="train",
    rank=0,
):
    if mode == "train":
        model.train()
        grad_env = torch.enable_grad
    elif mode == "test":
        model.eval()
        grad_env = torch.no_grad
    else:
        raise ValueError("This mode is not recognized.")


    kinematic_tree = loader.dataset.kinematic_tree
    offsets_sum = loader.dataset.offsets_sum
    
    with grad_env():
        epoch_metrics = {k: 0 for k in ["total"] + list(config["loss"].keys())}
        for batch_idx, data in enumerate(loader):
            # data = {
            #     k: v.to(device) if k != "offsets" else v[0].to(device)
            #     for k, v in data.items()
            # }

            # data["x3d"] = get_x3d_from_data(data, offsets_sum, kinematic_tree)

            if config["data"]["dataset"] == "mabe22":
                data = prepare_batch_2d_bespoke(
                    data=data,
                    augment_dict=config["train"]["augmentations"],
                    offsets_sum=offsets_sum,
                    device=device,
                )
            elif config["data"]["dataset"] in ["4mice", "parkinsons_healthy"]:
                data = prepare_batch(
                    data=data,
                    augment_dict=config["train"]["augmentations"],
                    offsets_sum=offsets_sum,
                    kinematic_tree=kinematic_tree,
                    device=device,
                    get_2d=config["model"]["is_2d"],
                )
            else:
                raise ValueError("Dataset not recognized.")
            
            data_o = predict_batch(model, data)

            # Unwrap DDP so losses.py can access model attributes like model.prior
            raw_model = model.module if isinstance(model, DDP) else model
            batch_loss = get_batch_loss(
                model=raw_model,
                data=data,
                data_o=data_o,
                loss_scale=config["loss"],
                kinematic_tree=kinematic_tree,
                offsets_sum=offsets_sum,
            )

            if mode == "train":
                # Update model parameters
                for param in model.parameters():
                    param.grad = None

                batch_loss["total"].backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1e5)
                optimizer.step()
                if scheduler is not None:
                    scheduler.step(epoch + batch_idx / len(loader))

            # Save loss and other metrics for the epoch
            epoch_metrics = {
                k: v + batch_loss[k].detach() for k, v in epoch_metrics.items()
            }

        # Calculate averages on all ranks
        for k, v in epoch_metrics.items():
            epoch_metrics[k] = v.item() / len(loader)
        
        # Only rank 0 prints
        if rank == 0:
            for k, v in epoch_metrics.items():
                print(
                    "====> Epoch: {} Average {} loss: {:.4f}".format(
                        epoch, k, epoch_metrics[k]
                    )
                )

    return epoch_metrics


def test_epoch(config, model, loader, device="cuda", epoch=0, rank=0):
    return train_test_epoch(
        config=config,
        model=model,
        loader=loader,
        optimizer=None,
        scheduler=None,
        device=device,
        epoch=epoch,
        mode="test",
        rank=rank,
    )


def train_epoch(config, model, loader, optimizer, scheduler, device="cuda", epoch=0, rank=0):
    return train_test_epoch(
        config=config,
        model=model,
        loader=loader,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        epoch=epoch,
        mode="train",
        rank=rank,
    )


# @profile
def train(config, model, loader_dict, run=None, rank=0):
    
    torch.set_float32_matmul_precision("medium")
    torch.autograd.set_detect_anomaly(True)
    torch.backends.cudnn.benchmark = True
    # config = balance_disentangle(config, loader_dict["train"].dataset)

    optimizer, scheduler = get_optimizer_and_lr_scheduler(
        model,
        config["train"],
        config["model"]["load_model"],
        config["model"]["start_epoch"],
    )

    # Apply beta annealing if selected
    if "prior" in config["loss"].keys():
        beta_scheduler = get_beta_schedule(
            config["loss"]["prior"],
            config["train"]["beta_anneal"],
        )
    else:
        beta_scheduler = None

    # Epoch loop
    for epoch in tqdm.trange(
        config["model"]["start_epoch"] + 1, config["train"]["num_epochs"] + 1
    ):
        # Set epoch on DistributedSampler for correct per-epoch shuffling
        if hasattr(loader_dict["train"].sampler, "set_epoch"):
            loader_dict["train"].sampler.set_epoch(epoch)

        # Update beta annealing if applicable
        if beta_scheduler is not None:
            config["loss"]["prior"] = beta_scheduler.get(epoch)
            if rank == 0:
                print("Beta schedule: {:.3f}".format(config["loss"]["prior"]))

        starttime = time.time()
        # Train for an epoch
        train_metrics = train_epoch(
            config=config,
            model=model,
            loader=loader_dict["train"],
            optimizer=optimizer,
            scheduler=scheduler,
            device="cuda",
            epoch=epoch,
            rank=rank,
        )
        metrics = {"{}_train".format(k): v for k, v in train_metrics.items()}
        metrics["time"] = time.time() - starttime

        # Unwrap DDP module for saving
        model_to_save = model.module if isinstance(model, DDP) else model

        if (rank == 0) and (epoch % 5 == 0) and (epoch > 1):
            if rank == 0:
                print("Saving model to folder: {}".format(config["out_path"]))
            torch.save(
                {k: v.cpu() for k, v in model_to_save.state_dict().items()},
                "{}/weights/epoch_{}.pth".format(config["out_path"], epoch),
            )

            if epoch % 20 == 0:
                torch.save(
                    {"optimizer": optimizer.state_dict(), "lr_scheduler": scheduler},
                    "{}/checkpoints/epoch_{}.pth".format(config["out_path"], epoch),
                )

            ## Calculate test metrics (rank 0 only)
            if epoch >= 50:
                config_test = copy.deepcopy(config)
                config_test["train"]["augmentations"] = {
                    "2d_td": None,
                    "kpt_shuffle": None,
                    "offset_noise": None,
                    "single_ablation": None,
                    "temporal_mask_past": None,
                    "temporal_mask_future": None,
                    "temporal_mask_random": None,
                }
                # Round 1
                test_metrics = test_epoch(
                    config=config_test,
                    model=model,
                    loader=loader_dict["val"],
                    device="cuda",
                    epoch=epoch,
                    rank=rank,
                )

                metrics.update(
                    {"{}_test".format(k): v for k, v in test_metrics.items()}
                )

                config_test["loss"] = {"jpe": 1, "root": 1}

                # Round 2
                config_test["train"]["augmentations"] = {
                    "2d_td": None,
                    "kpt_shuffle": True,
                    "offset_noise": None,
                    "single_ablation": None,
                    "temporal_mask_past": None,
                    "temporal_mask_future": None,
                    "temporal_mask_random": None,
                }
                test_metrics = test_epoch(
                    config=config_test,
                    model=model,
                    loader=loader_dict["val"],
                    device="cuda",
                    epoch=epoch,
                    rank=rank,
                )
                metrics.update(
                    {"{}_test_kpt_shuffle".format(k): v for k, v in test_metrics.items()}
                )

                # Round 3
                config_test["train"]["augmentations"] = {
                    "2d_td": 1.0,
                    "kpt_shuffle": None,
                    "offset_noise": None,
                    "single_ablation": None,
                    "temporal_mask_past": None,
                    "temporal_mask_future": None,
                    "temporal_mask_random": None,
                }
                test_metrics = test_epoch(
                    config=config_test,
                    model=model,
                    loader=loader_dict["val"],
                    device="cuda",
                    epoch=epoch,
                    rank=rank,
                )
                metrics.update(
                    {"{}_test_2d_td".format(k): v for k, v in test_metrics.items()}
                )

                # Round 4
                config_test["train"]["augmentations"] = {
                    "2d_td": None,
                    "kpt_shuffle": None,
                    "offset_noise": None,
                    "single_ablation": 1.0,
                    "temporal_mask_past": None,
                    "temporal_mask_future": None,
                    "temporal_mask_random": None,
                }
                test_metrics = test_epoch(
                    config=config_test,
                    model=model,
                    loader=loader_dict["val"],
                    device="cuda",
                    epoch=epoch,
                    rank=rank,
                )
                metrics.update(
                    {"{}_test_single_ablation".format(k): v for k, v in test_metrics.items()}
                )

        # All ranks barrier here - wait for rank 0 to finish checkpointing and validation
        if torch.distributed.is_initialized():
            torch.distributed.barrier()

        if run is not None:
            wandb.log(metrics, epoch)

    return model


# /hpc/home/yw789/tdunn/befound_code/befound/src/befound/train/trainer.py