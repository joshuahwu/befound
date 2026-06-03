
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
    get_optimizer_and_lr_scheduler,
    get_beta_schedule,
    train_epoch,
    test_epoch,
)

class FormatAdaptedCIResVAE(CIResVAE2D):
    """
    Box-pretrained CIResVAE backbone + lightweight per-format linear adapters.

    Inherits CIResVAE2D so isinstance(model, CIResVAE2D) is True, which routes
    get_batch_loss to the 2D loss path (jpe + root + prior).

    Only input_proj and output_proj are newly initialised; backbone conv kernels
    come from the box checkpoint (loaded strict=True) and are fully trainable
    during fine-tuning.

    Forgetting check: pass model.backbone directly to test_epoch with the box
    config (dataset=parkinsons_healthy).  backbone is CIResVAE, so isinstance
    checks are False → 3D loss path, all three 3D losses valid.
    """

    def __init__(self, backbone: CIResVAE, d_in: int = 2, n_kpt: int = 12):
        nn.Module.__init__(self)           # bypass CIResVAE2D.__init__
        self.prior       = backbone.prior
        self.dist_params = backbone.dist_params
        self.backbone    = backbone
        self.n_keypts    = n_kpt
        self.window_size = backbone.window_size
        self.hidden_dim  = backbone.hidden_dim
        self._d_in       = d_in

        # per-keypoint 2D→3D linear projection (Conv1d over time, kernel_size=1)
        # initialised so first d_in output channels are an identity copy; d_in+1..3 = 0
        # self.input_proj = nn.Conv1d(d_in, backbone.in_channels, kernel_size=1, bias=False)
        # with torch.no_grad():
        #     nn.init.zeros_(self.input_proj.weight)
        #     self.input_proj.weight[:d_in, :, 0] = torch.eye(d_in)

        # backbone decoder produces out_channels=111 features; project to K*d_in
        self.output_proj = nn.Linear(backbone.out_channels, n_kpt * d_in)

    def encode(self, data: dict) -> dict:
        x2d = data["x2d"]                                          # (B, W, K, d_in)
        B, W, K, D = x2d.shape
        # x_flat = x.permute(0, 2, 3, 1).reshape(B * K, D, W)     # (B*K, d_in, W)
        # x_proj = self.input_proj(x_flat)                         # (B*K, 3, W)
        # x_4d   = x_proj.reshape(B, K, self.backbone.in_channels, W)   # (B, K, 3, W)
        x_3d = torch.cat([x2d, torch.zeros_like(x2d[..., :1])], dim=-1)  # (B, W, K, 3)
        x_4d = x_3d.permute(0, 2, 3, 1)  # (B, K, 3, W)
        out    = self.backbone.encoder(x_4d)                     # (B, n_ch_out, latent_T)
        out_flat = out.flatten(1)                                 # (B, n_ch_out * latent_T)
        return {
            "mu":     self.backbone.fc_mu(out_flat),
            "logvar": self.backbone.fc_logvar(out_flat),
        }

    def decode(self, z: torch.Tensor) -> dict:
        B   = z.shape[0]
        out = self.backbone.fc_decoder(z)                              # (B, latent_T*hidden[-1])
        out = out.reshape(B, self.backbone.hidden_dim[-1], -1)         # (B, hidden[-1], latent_T)
        raw = self.backbone.decoder(out).permute(0, 2, 1)              # (B, W, out_channels=111)
        x2d = self.output_proj(raw).reshape(                           # (B, W, K, d_in)
            B, self.window_size, self.n_keypts, self._d_in
        )
        return {"x2d": x2d}

parser = argparse.ArgumentParser(
    description="Fine-tune box-pretrained CI-VAE on MABe22 via format adapters"
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

model = FormatAdaptedCIResVAE(backbone=backbone, d_in=2, n_kpt=12).cuda()
print("FormatAdaptedCIResVAE ready — n_kpt={}, d_in={}, window={}".format(
    model.n_keypts, model._d_in, model.window_size
))

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

optimizer, scheduler = get_optimizer_and_lr_scheduler(
    model,
    config["train"],
    load_path=None,    # fresh optimizer — do not resume box-training state
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

# Box forgetting evaluation — passes model.backbone (CIResVAE) to test_epoch,
# which routes to the 3D loss path via isinstance checks.
_config_box_eval = copy.deepcopy(config)
_config_box_eval["data"]["dataset"]        = "parkinsons_healthy"
_config_box_eval["loss"]                   = {"jpe": 1, "root": 1, "prior": 1}
_config_box_eval["train"]["augmentations"] = _NO_AUGMENT

for epoch in tqdm.trange(1, config["train"]["num_epochs"] + 1):
    if hasattr(loader_dict["train"].sampler, "set_epoch"):
        loader_dict["train"].sampler.set_epoch(epoch)

    if beta_scheduler is not None:
        config["loss"]["prior"] = beta_scheduler.get(epoch)
        print("Beta schedule: {:.3f}".format(config["loss"]["prior"]))

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

    if epoch % 5 == 0 and epoch > 1:
        print("Saving weights at epoch {}".format(epoch))
        torch.save(
            {k: v.cpu() for k, v in model.state_dict().items()},
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

            # MABe22 submission-split validation: 2D jpe + root + prior
            mabe_metrics = test_epoch(
                config=config_test,
                model=model,
                loader=loader_dict["val"],
                device="cuda",
                epoch=epoch,
            )
            metrics.update({f"{k}_mabe_val": v for k, v in mabe_metrics.items()})

            # Box forgetting probe: backbone only, 3D loss path
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

# python finetune_mabe22.py -p ci_vae/finetune -n mabe22_ft \
#     --out_path /hpc/group/tdunn/yw789/befound_results/
