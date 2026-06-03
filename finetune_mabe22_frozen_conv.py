

from __future__ import annotations

import argparse
import copy
import time
from pathlib import Path

import torch
import torch.nn as nn
import tqdm
import wandb
from torch.utils.data import DataLoader

from befound.params import read as params_read
from befound.get.get import get_model, get_mouse_data, get_mabe22_data
from befound.data import MabeWindowDataset
from befound.model.vae import CIResVAE, CIResVAE2D
from befound.train.trainer import (
    get_beta_schedule,
    train_epoch,
    test_epoch,
)


# Dual-head model
class DualOutCIResVAE(CIResVAE2D):
    """
    Backbone decoder is intercepted just before its final Conv1d+tanh.
    Two parallel heads read from the same (B, out_channels, T') feature map:

      backbone.decoder.out  : Conv1d(out_channels→out_channels, k=out_k) — frozen, box
      out_mabe              : Conv1d(out_channels→n_kpt*d_in, k=out_k)   — trained, MABe

    Inherits CIResVAE2D so get_batch_loss uses the 2D loss path.
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

        out_k = backbone.decoder.out.kernel_size[0]   # mirrors backbone.decoder.out exactly
        self.out_mabe = nn.Conv1d(
            backbone.out_channels,   # 111
            n_kpt * d_in,            # 24
            kernel_size=out_k,       # 5
            stride=1,
            padding=0,
        )

    def encode(self, data: dict) -> dict:
        x2d = data["x2d"]                                          # (B, W, K, d_in)
        B, W, K, D = x2d.shape
        x_3d = torch.cat([x2d, torch.zeros_like(x2d[..., :1])], dim=-1)  # (B, W, K, 3)
        x_4d = x_3d.permute(0, 2, 3, 1)                           # (B, K, 3, W)
        out     = self.backbone.encoder(x_4d)                      # (B, n_ch_out, latent_T)
        out_flat = out.flatten(1)
        return {
            "mu":     self.backbone.fc_mu(out_flat),
            "logvar": self.backbone.fc_logvar(out_flat),
        }

    def decode(self, z: torch.Tensor) -> dict:
        B   = z.shape[0]
        out = self.backbone.fc_decoder(z)
        out = out.reshape(B, self.backbone.hidden_dim[-1], -1)
        # Run decoder backbone (res+upsamp blocks) but skip the final out conv
        out = self.backbone.decoder.input(out)
        out = self.backbone.decoder.backbone(out)   # (B, out_channels=111, T'=55)
        # MABe head — parallel to frozen backbone.decoder.out
        x2d = torch.tanh(self.out_mabe(out))        # (B, 24, 51)
        x2d = x2d.permute(0, 2, 1).reshape(
            B, self.window_size, self.n_keypts, self._d_in
        )
        return {"x2d": x2d}

parser = argparse.ArgumentParser(
    description="Fine-tune CI-VAE on MABe22 — frozen backbone, parallel out_mabe conv"
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

wandb.login()

config = params_read.config(
    "{}/{}/{}/model_config.yaml".format(args.out_path, args.project, args.name)
)

run = wandb.init(
    project=args.project.replace("/", "_"),
    name=args.name,
    config=config,
    dir=config["out_path"],
)
print("WANDB directory: {}".format(run.dir))


box_loader = get_mouse_data(
    data_config=config["box_eval"],
    train_val_test="val",
    data_keys=["x6d", "root", "offsets", "target_pose"],
    shuffle=False,
    use_default_offsets=True,
)
box_n_keypts = box_loader.dataset.n_keypts
print("Box val dataset — n_keypts={}, offsets_sum={:.1f}".format(
    box_n_keypts, box_loader.dataset.offsets_sum
))

backbone = get_model(
    model_config=config["model"],
    load_model=None,
    epoch=None,
    n_keypts=box_n_keypts,
    device="cuda",
    verbose=1,
)

load_path = "{}/weights/epoch_{}.pth".format(
    config["model"]["load_model"], config["model"]["start_epoch"]
)
print("Loading pretrained weights (strict=True) from:\n{}".format(load_path))
state_dict = torch.load(load_path, map_location="cpu")
missing_keys, unexpected_keys = backbone.load_state_dict(state_dict, strict=True)
print("Missing keys   :", missing_keys)
print("Unexpected keys:", unexpected_keys)

pretrain_epoch = config["model"]["start_epoch"]
config["model"]["start_epoch"] = 0
print("Pretrained from epoch {}; resetting epoch counter to 0.".format(pretrain_epoch))

model = DualOutCIResVAE(backbone=backbone, d_in=2, n_kpt=12).cuda()

# Freeze entire backbone
for param in model.backbone.parameters():
    param.requires_grad = False

trainable = [(n, p.shape) for n, p in model.named_parameters() if p.requires_grad]
n_frozen    = sum(p.numel() for p in model.parameters() if not p.requires_grad)
n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
print("\nFrozen parameters  : {:,}".format(n_frozen))
print("Trainable parameters: {:,}".format(n_trainable))
for n, s in trainable:
    print("  {} {}".format(n, list(s)))

mabe_path   = config["data"]["data_path"]
window      = config["model"]["window"]
batch_size  = config["data"]["batch_size"]
num_workers = 2

kp_train, _ = get_mabe22_data(mabe_path, split="train", reindex=True)
ds_train     = MabeWindowDataset(kp_train, window=window, stride=6,  pad_mode="edge")
loader_train = DataLoader(
    ds_train, batch_size=batch_size, shuffle=True,
    num_workers=num_workers, pin_memory=True, drop_last=True,
    persistent_workers=num_workers > 0,
)

kp_val, _ = get_mabe22_data(mabe_path, split="submission", reindex=True)
ds_val     = MabeWindowDataset(kp_val, window=window, stride=20, pad_mode="edge")
loader_val = DataLoader(
    ds_val, batch_size=batch_size, shuffle=False,
    num_workers=num_workers, pin_memory=True, drop_last=False,
    persistent_workers=num_workers > 0,
)

loader_dict = {"train": loader_train, "val": loader_val}
print("MABe22 train windows: {}  val windows: {}".format(len(ds_train), len(ds_val)))


torch.set_float32_matmul_precision("medium")
torch.backends.cudnn.benchmark = True

lr        = config["train"]["lr"]
optimizer = torch.optim.AdamW(model.out_mabe.parameters(), lr=lr)
scheduler = None
print("Optimizer: AdamW  lr={:.1e}  constant (no schedule)".format(lr))

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

for epoch in tqdm.trange(1, config["train"]["num_epochs"] + 1):
    if beta_scheduler is not None:
        config["loss"]["prior"] = beta_scheduler.get(epoch)

    t0 = time.time()
    train_metrics = train_epoch(
        config=config,
        model=model,
        loader=loader_dict["train"],
        optimizer=optimizer,
        scheduler=scheduler,
        device="cuda",
        epoch=epoch,
    )
    metrics = {f"{k}_train": v for k, v in train_metrics.items()}
    metrics["time"] = time.time() - t0
    metrics["lr"]   = lr

    if epoch % 5 == 0 and epoch > 1:
        print("Saving weights at epoch {}".format(epoch))
        torch.save(
            {k: v.cpu() for k, v in model.state_dict().items()},
            "{}/weights/epoch_{}.pth".format(config["out_path"], epoch),
        )

        if epoch % 20 == 0:
            torch.save(
                {"optimizer": optimizer.state_dict(), "lr_scheduler": None},
                "{}/checkpoints/epoch_{}.pth".format(config["out_path"], epoch),
            )

        if epoch >= 20:
            config_test = copy.deepcopy(config)
            config_test["train"]["augmentations"] = _NO_AUGMENT

            mabe_metrics = test_epoch(
                config=config_test,
                model=model,
                loader=loader_dict["val"],
                device="cuda",
                epoch=epoch,
            )
            metrics.update({f"{k}_mabe_val": v for k, v in mabe_metrics.items()})

            # Box forgetting probe — should stay flat since backbone is frozen
            box_metrics = test_epoch(
                config=_config_box_eval,
                model=model.backbone,
                loader=box_loader,
                device="cuda",
                epoch=epoch,
            )
            metrics.update({f"{k}_box_val": v for k, v in box_metrics.items()})
            print(
                "  box_jpe={:.4f}  box_root={:.4f}  box_prior={:.4f}  box_total={:.4f}".format(
                    box_metrics.get("jpe", float("nan")),
                    box_metrics.get("root", float("nan")),
                    box_metrics.get("prior", float("nan")),
                    box_metrics.get("total", float("nan")),
                )
            )

    wandb.log(metrics, epoch)

run.finish()
