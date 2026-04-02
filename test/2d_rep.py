import befound
from befound.params import read
import torch
from befound.data.train_utils import prepare_batch
import numpy as np
from pathlib import Path
import copy
from scipy.spatial import procrustes
import matplotlib.pyplot as plt

RESULTS_PATH = "/mnt/home/jwu10/working/ceph/results/foundation/"
key_dict = {
    "f_vae/vae_2d": [380, ["2d"]],
    "f_vae/2d_td": [400, ["2d", "3d"]],
    "f_vae/vae": [400, ["2d", "3d"]],
    "ci_vae/ci_vae_2d": [265, ["2d"]],
    "ci_vae/2d_td": [400, ["2d", "3d"]],
    "ci_vae/ci_vae": [400, ["2d","3d"]],
}

z_dict = {}
for analysis_key, (load_epoch, dim_key) in key_dict.items():
    config = read.config("{}/{}/model_config.yaml".format(RESULTS_PATH, analysis_key))
    config["model"]["load_model"] = config["out_path"]
    config["model"]["start_epoch"] = load_epoch

    loader_dict, model = None, None
    for dim_k in dim_key:
        latent_path = Path(
            f'{config["out_path"]}/latents/z_{dim_k}_{config["model"]["start_epoch"]}.npy'
        )
        if latent_path.is_file():
            z_dict[f"{analysis_key}_{dim_k}"] = np.load(latent_path)
        else:
            if None in [model, loader_dict]:
                loader_dict, model = befound.get.data_and_model(
                    config,
                    train_val_test=["val"],
                    data_keys=["x6d", "root", "offsets", "target_pose"],
                    shuffle=[False],
                    use_default_offsets=[True],
                )
            loader = loader_dict["val"]
            offsets_sum = loader.dataset.offsets_sum
            kinematic_tree = loader.dataset.kinematic_tree

            if (dim_k == "2d") and ("vae_2d" not in analysis_key):
                augment_dict = {
                    "2d_td": 1,
                    "kpt_shuffle": None,
                    "offset_noise": None,
                    "single_ablation": None,
                }
            else:
                augment_dict = {
                    "2d_td": None,
                    "kpt_shuffle": None,
                    "offset_noise": None,
                    "single_ablation": None,
                }

            model.eval()
            # for k, v in augment_dicts.items():
            z = []
            with torch.no_grad():
                for batch_idx, data in enumerate(loader):
                    data = {k: v.cuda() for k, v in data.items()}
                    data = prepare_batch(
                        data=data,
                        augment_dict=augment_dict,
                        offsets_sum=offsets_sum,
                        kinematic_tree=kinematic_tree,
                        device="cuda",
                        get_2d=config["model"]["is_2d"],
                    )

                    z += [model.encode(data)["mu"].detach()]

                z = torch.cat(z, dim=0).cpu().numpy()
                np.save(latent_path, z)
            z_dict[f"{analysis_key}_{dim_k}"] = copy.deepcopy(z)

z_np = np.concatenate([v[None, ...] for k,v in z_dict.items()],axis=0)

procrustes_mat = np.zeros((z_np.shape[0],)*2)

for i in range(z_np.shape[0]):
    for j in range(i+1, z_np.shape[0]):
        procrustes_mat[i,j] = procrustes(z_np[i], z_np[j])[2]

procrustes_mat += procrustes_mat.T

f = plt.figure()
plt.imshow(procrustes_mat)
plt.xticks(np.arange(z_np.shape[0]), list(z_dict.keys()), rotation=90)
plt.yticks(np.arange(z_np.shape[0]),list(z_dict.keys()))
plt.colorbar()
f.tight_layout()
plt.savefig("./test_procrustes.png")
plt.close()
import pdb

pdb.set_trace()
