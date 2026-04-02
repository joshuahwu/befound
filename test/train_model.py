import scrubvae
from pathlib import Path
from scrubvae.params import read
import wandb
import argparse
import model
from neuroposelib import vis
import data
import neuroposelib as npl


# Read in config file with all parameters and settings
config = read.config("/mnt/home/jwu10/working/vqmap_testing/test/model_config.yaml")
connectivity = npl.read.connectivity_config(
    config["data"]["data_path"] + "mouse_skeleton.yaml"
)

# Get DataLoaders and model
dataset_label = "val"
loader = data.get_mouse_data(
    data_config=config["data"],
    train_val_test=dataset_label,
    data_keys=["x3d"],
    shuffle=False,
)

# vis.pose.arena3D(
#     loader.dataset[0]["x3d"].numpy(),
#     connectivity,
#     frames=[0],
#     centered=False,
#     fps=30,
#     N_FRAMES=51,
#     VID_NAME="test.mp4",
#     SAVE_ROOT="./",
# # )

# channel_encoder = model.conv.Conv1DEncoder(
#     in_channels=3,
#     hidden_dim=[128, 256, 512],
#     strides=[2, 2, 2],
#     depth=2,
#     dilation=1,
#     activation="prelu",
#     normalization="LN",
#     out_kernel_size=3,
# ).cuda()

# # test_out = channel_encoder(data["x3d"].permute(0, 2, 3, 1))

# vqmap_encoder = model.vae.ChannelInvariantEncoder(
#     channel_encoder=channel_encoder,
#     num_heads=4,
#     query_size=8,
#     chan_mix_last=False,
#     num_points=None,
#     latent_dim=64,
#     use_be=True,
# ).cuda()

data = {k: v.cuda() for k, v in loader.dataset[:128].items()}

# test_out = vqmap_encoder(data["x3d"].permute(0, 2, 3, 1))
# decoder = model.conv.Conv1DDecoder(
#     out_channels=3,
#     hidden_dim=[512, 256, 128],
#     strides=[2, 2, 2],
#     depth=2,
#     dilation=1,
#     activation="prelu",
#     normalization="LN",
#     out_kernel_size=3,
# ).cuda()

rvae = model.vae.ResVAE(
    in_channels=3,
    hidden_dim=[128, 256, 512],
    latent_dim=64,
    depth=2,
    query_size=8,
    activation="prelu",
    out_kernel_size=4,
    prior="gaussian",
).cuda()

test_out = rvae(data)

import pdb

pdb.set_trace()
