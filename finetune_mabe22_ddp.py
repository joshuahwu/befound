

from __future__ import annotations

import argparse
import copy
import datetime
import os
import time
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn as nn
import tqdm
import wandb
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from befound.params import read as params_read
from befound.get.get import get_model, get_mouse_data, get_mabe22_data
from befound.data import MabeWindowDataset
from befound.model.vae import CIResVAE, CIResVAE2D
from befound.train.trainer import (
    get_optimizer_and_lr_scheduler,
    get_beta_schedule,
    train_epoch,
    test_epoch,
)


def setup_ddp():
    dist.init_process_group(
        backend="nccl",
        timeout=datetime.timedelta(minutes=60),
    )
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank, dist.get_rank(), dist.get_world_size()


def cleanup_ddp():
    dist.destroy_process_group()

class FormatAdaptedCIResVAE(CIResVAE2D):
    """
    Box backbone + MABe22 output adapter.  Inherits CIResVAE2D for 2D loss path.
    backbone.encoder is channel-invariant (K=12 works through K=18 kernels).
    Forgetting check: access via model.module.backbone (DDP-unwrapped).
    """

    def __init__(self, backbone: CIResVAE, d_in: int = 2, n_kpt: int = 12):
        nn.Module.__init__(self)
        self.prior       = backbone.prior
        self.dist_params = backbone.dist_params
        self.backbone    = backbone
        self.n_keypts    = n_kpt
        self.window_size = backbone.window_size
        self.hidden_dim  = backbone.hidden_dim
        self._d_in       = d_in
        self.output_proj = nn.Linear(backbone.out_channels, n_kpt * d_in)

    def encode(self, data: dict) -> dict:
        x2d  = data["x2d"]                                          # (B, W, K, 2)
        B, W, K, D = x2d.shape
        x_3d = torch.cat([x2d, torch.zeros_like(x2d[..., :1])], dim=-1)  # (B, W, K, 3)
        x_4d = x_3d.permute(0, 2, 3, 1)                            # (B, K, 3, W)
        out  = self.backbone.encoder(x_4d)                          # (B, n_ch_out, latent_T)
        out_flat = out.flatten(1)                                    # (B, n_ch_out * latent_T)
        return {
            "mu":     self.backbone.fc_mu(out_flat),
            "logvar": self.backbone.fc_logvar(out_flat),
        }

    def decode(self, z: torch.Tensor) -> dict:
        B   = z.shape[0]
        out = self.backbone.fc_decoder(z)
        out = out.reshape(B, self.backbone.hidden_dim[-1], -1)
        raw = self.backbone.decoder(out).permute(0, 2, 1)           # (B, W, 111)
        x2d = self.output_proj(raw).reshape(
            B, self.window_size, self.n_keypts, self._d_in
        )
        return {"x2d": x2d}


def main():
    local_rank, rank, world_size = setup_ddp()

    parser = argparse.ArgumentParser(
        description="Fine-tune box CI-VAE on MABe22 (DDP)"
    )
    parser.add_argument("--out_path", "-o", type=str, dest="out_path", default=None)
    parser.add_argument("--job_id",   type=int, dest="job_id",   default=None)
    parser.add_argument("--project",  "-p", type=str, dest="project",  required=True)
    parser.add_argument("--name",     "-n", type=str, dest="name",     default=None)
    args = parser.parse_args()

    if args.out_path is None:
        args.out_path = "/hpc/group/tdunn/yw789/befound_results/"

    if args.job_id is not None:
        z_path = Path(args.out_path + args.project)
        folders = sorted([str(f.parts[-1]) for f in z_path.iterdir() if f.is_dir()])
        args.name = folders[args.job_id]

    if rank == 0:
        wandb.login()

    config = params_read.config(
        "{}/{}/{}/model_config.yaml".format(args.out_path, args.project, args.name)
    )

    if rank == 0:
        run = wandb.init(
            project=args.project.replace("/", "_"),
            name=args.name,
            config=config,
            dir=config["out_path"],
        )
        print("WANDB directory: {}".format(run.dir))
    else:
        run = None

    box_loader = get_mouse_data(
        data_config=config["box_eval"],
        train_val_test="val",
        data_keys=["x6d", "root", "offsets", "target_pose"],
        shuffle=False,
        use_default_offsets=False,
    )
    box_n_keypts = box_loader.dataset.n_keypts
    if rank == 0:
        print("Box val dataset — n_keypts={}, offsets_sum={:.1f}".format(
            box_n_keypts, box_loader.dataset.offsets_sum
        ))

    backbone = get_model(
        model_config=config["model"],
        load_model=None,
        epoch=None,
        n_keypts=box_n_keypts,
        device=f"cuda:{local_rank}",
        verbose=(1 if rank == 0 else 0),
    )

    load_path = "{}/weights/epoch_{}.pth".format(
        config["model"]["load_model"], config["model"]["start_epoch"]
    )
    if rank == 0:
        print("Loading pretrained weights (strict=True) from:\n{}".format(load_path))
    state_dict = torch.load(load_path, map_location="cpu")
    missing_keys, unexpected_keys = backbone.load_state_dict(state_dict, strict=True)
    if rank == 0:
        print("Missing keys   :", missing_keys)
        print("Unexpected keys:", unexpected_keys)

    pretrain_epoch = config["model"]["start_epoch"]
    config["model"]["start_epoch"] = 0


    model = FormatAdaptedCIResVAE(backbone=backbone, d_in=2, n_kpt=12)
    model = model.to(f"cuda:{local_rank}")
    # find_unused_parameters=True: backbone.encoder.be is never used (pe_indices=None)
    model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)
    if rank == 0:
        print("FormatAdaptedCIResVAE (DDP) — n_kpt=12, window={}".format(
            model.module.window_size
        ))


    mabe_path  = config["data"]["data_path"]
    window     = config["model"]["window"]
    batch_size = config["data"]["batch_size"]
    num_workers = 2

    kp_train, _ = get_mabe22_data(mabe_path, split="train", reindex=True)
    ds_train     = MabeWindowDataset(kp_train, window=window, stride=6,  pad_mode="edge")
    sampler_train = DistributedSampler(
        ds_train, num_replicas=world_size, rank=rank, shuffle=True
    )
    loader_train = DataLoader(
        ds_train, batch_size=batch_size, sampler=sampler_train,
        num_workers=num_workers, pin_memory=True, drop_last=True,
    )

    kp_val, _ = get_mabe22_data(mabe_path, split="submission", reindex=True)
    ds_val     = MabeWindowDataset(kp_val, window=window, stride=20, pad_mode="edge")
    sampler_val = DistributedSampler(
        ds_val, num_replicas=world_size, rank=rank, shuffle=False
    )
    loader_val = DataLoader(
        ds_val, batch_size=batch_size, sampler=sampler_val,
        num_workers=num_workers, pin_memory=True, drop_last=False,
    )

    loader_dict = {"train": loader_train, "val": loader_val}
    if rank == 0:
        print("MABe22 train windows: {}  val windows: {}".format(
            len(ds_train), len(ds_val)
        ))


    torch.set_float32_matmul_precision("medium")
    torch.backends.cudnn.benchmark = True

    optimizer, scheduler = get_optimizer_and_lr_scheduler(
        model,
        config["train"],
        load_path=None,
        start_epoch=None,
    )

    beta_scheduler = None
    if "prior" in config["loss"]:
        beta_scheduler = get_beta_schedule(
            config["loss"]["prior"],
            config["train"]["beta_anneal"],
        )

    _NO_AUGMENT = {
        "2d_td": None,
        "kpt_shuffle": None,
        "offset_noise": None,
        "single_ablation": None,
        "temporal_mask_past": None,
        "temporal_mask_future": None,
        "temporal_mask_random": None,
    }

    _config_box_eval = copy.deepcopy(config)
    _config_box_eval["data"]["dataset"]        = "parkinsons_healthy"
    _config_box_eval["loss"]                   = {"jpe": 1, "root": 1, "prior": 1}
    _config_box_eval["train"]["augmentations"] = _NO_AUGMENT

    def _run_box_val(epoch):
        """Run box forgetting probe on rank 0; return metrics dict."""
        box_metrics = test_epoch(
            config=_config_box_eval,
            model=model.module.backbone,
            loader=box_loader,
            device=f"cuda:{local_rank}",
            epoch=epoch,
            rank=rank,
        )
        print(
            "  [epoch {}] box_jpe={:.4f}  box_root={:.4f}  box_prior={:.4f}  box_total={:.4f}".format(
                epoch,
                box_metrics.get("jpe", float("nan")),
                box_metrics.get("root", float("nan")),
                box_metrics.get("prior", float("nan")),
                box_metrics.get("total", float("nan")),
            )
        )
        return box_metrics

    if rank == 0:
        print("Running box val baseline before training ...")
        box_metrics_0 = _run_box_val(epoch=0)
        if run is not None:
            wandb.log({f"{k}_box_val": v for k, v in box_metrics_0.items()}, 0)
    dist.barrier()

    for epoch in tqdm.trange(
        1, config["train"]["num_epochs"] + 1, disable=(rank != 0)
    ):
        sampler_train.set_epoch(epoch)

        if beta_scheduler is not None:
            config["loss"]["prior"] = beta_scheduler.get(epoch)
            if rank == 0:
                print("Beta schedule: {:.3f}".format(config["loss"]["prior"]))

        t0 = time.time()
        train_metrics = train_epoch(
            config=config,
            model=model,
            loader=loader_dict["train"],
            optimizer=optimizer,
            scheduler=scheduler,
            device=f"cuda:{local_rank}",
            epoch=epoch,
            rank=rank,
        )
        metrics = {f"{k}_train": v for k, v in train_metrics.items()}
        metrics["time"] = time.time() - t0

        if rank == 0 and epoch % 5 == 0:
            # Box val every 5 epochs (no epoch >= 50 gate)
            box_metrics = _run_box_val(epoch)
            metrics.update({f"{k}_box_val": v for k, v in box_metrics.items()})

            if epoch > 1:
                print("Saving weights at epoch {}".format(epoch))
                torch.save(
                    {k: v.cpu() for k, v in model.module.state_dict().items()},
                    "{}/weights/epoch_{}.pth".format(config["out_path"], epoch),
                )

                if epoch % 20 == 0:
                    torch.save(
                        {"optimizer": optimizer.state_dict(), "lr_scheduler": scheduler},
                        "{}/checkpoints/epoch_{}.pth".format(config["out_path"], epoch),
                    )

            if epoch >= 50:
                config_test = copy.deepcopy(config)
                config_test["train"]["augmentations"] = _NO_AUGMENT

                # MABe22 val: 2D jpe + root + prior (DDP model, rank-0 loader)
                mabe_metrics = test_epoch(
                    config=config_test,
                    model=model,
                    loader=loader_dict["val"],
                    device=f"cuda:{local_rank}",
                    epoch=epoch,
                    rank=rank,
                )
                metrics.update({f"{k}_mabe_val": v for k, v in mabe_metrics.items()})

        # sync all ranks before next epoch
        dist.barrier()

        if rank == 0 and run is not None:
            wandb.log(metrics, epoch)

    if rank == 0:
        run.finish()

    cleanup_ddp()


if __name__ == "__main__":
    main()

# torchrun --nproc_per_node=2 finetune_mabe22_ddp.py \
#     -p ci_vae/finetune -n mabe22_ft \
#     --out_path /hpc/group/tdunn/yw789/befound_results/
