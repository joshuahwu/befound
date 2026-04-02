import befound
from neuroposelib import read
import torch
from befound.data.train_utils import prepare_batch
import numpy as np
from scrubvae.data.dataset import fwd_kin_cont6d_torch
from neuroposelib import vis

RESULTS_PATH = "/mnt/home/jwu10/working/ceph/results/foundation/"
analysis_key = "f_vae/vae_2d"

config = read.config("{}/{}/model_config.yaml".format(RESULTS_PATH, analysis_key))
config["model"]["load_model"] = config["out_path"]
config["model"]["start_epoch"] = 315
connectivity = read.connectivity_config(
    config["data"]["data_path"] + "mouse_skeleton.yaml"
)

loader_dict, model = befound.get.data_and_model(
    config,
    train_val_test=["val"],
    data_keys=["x6d", "root", "offsets", "target_pose"],
    shuffle=[True, False],
    use_default_offsets=[True, False],
)
loader = loader_dict["val"]
offsets_sum = loader.dataset.offsets_sum
window = config["model"]["window"]
n_keypts = loader.dataset.n_keypts
kinematic_tree = loader.dataset.kinematic_tree

augment_dict = {
    "2d_td": None,
    "kpt_shuffle": None,
    "offset_noise": None,
    "single_ablation": None,
}

model.eval()
sample_inds = [10000, 100000, 253405]
z = []
data = loader.dataset[sample_inds]
data = prepare_batch(
    data=data,
    augment_dict=augment_dict,
    offsets_sum=offsets_sum,
    kinematic_tree=kinematic_tree,
    device="cuda",
    get_2d=config["model"]["is_2d"],
)
data_o = model(data)

# pose = fwd_kin_cont6d_torch(
#         data_o["x6d"].reshape((-1, n_keypts, 6)),
#         kinematic_tree,
#         offsets.reshape((-1, n_keypts, 3)),
#         root_pos=data_o["root"].reshape(-1, 3),
#         do_root_R=True,
#     ).reshape(-1, window, n_keypts, 3)

pose = np.zeros((*data_o["x2d"].shape[:-1], 3)) + 0.01
pose[..., :2] = data_o["x2d"].cpu().detach().numpy()
vis.pose.arena3D(
    pose.reshape(-1, n_keypts, 3) * offsets_sum,
    connectivity,
    frames=[0],
    centered=False,
    fps=30,
    N_FRAMES=window * len(sample_inds),
    VID_NAME=f"test_2d_o.mp4",
    SAVE_ROOT="./",
)

pose = np.zeros((*data["x2d"].shape[:-1], 3)) + 0.01
pose[..., :2] = data["x2d"].cpu().detach().numpy()
vis.pose.arena3D(
    pose.reshape(-1, n_keypts, 3) * offsets_sum,
    connectivity,
    frames=[0],
    centered=False,
    fps=30,
    N_FRAMES=window * len(sample_inds),
    VID_NAME=f"test_2d.mp4",
    SAVE_ROOT="./",
)

pose = np.zeros((*data["target_pose"].shape[:-1], 3)) + 0.01
pose[..., :2] = data["target_pose"][..., :2].cpu().detach().numpy()
vis.pose.arena3D(
    pose.reshape(-1, n_keypts, 3) * offsets_sum,
    connectivity,
    frames=[0],
    centered=False,
    fps=30,
    N_FRAMES=window * len(sample_inds),
    VID_NAME=f"test_2d_target.mp4",
    SAVE_ROOT="./",
)

import pdb

pdb.set_trace()
