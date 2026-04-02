import befound
from befound.params import read
from pathlib import Path
import wandb
import argparse

# argparse project and job names
parser = argparse.ArgumentParser(prog="Behavioral Foundation Modeles Train", description="Train Behavioral Foundation Models")
parser.add_argument("--out_path", "-o", type=str, dest="out_path")
parser.add_argument("--job_id", type=int, dest="job_id")
parser.add_argument("--project", "-p", type=str, dest="project")
parser.add_argument("--name", "-n", type=str, dest="name")
args = parser.parse_args()

if args.out_path is None:
    args.out_path = "/mnt/home/jwu10/working/ceph/results/foundation/"

### Set/Load Parameters
wandb.login()
if args.job_id is not None:
    z_path = Path(args.out_path + args.project)
    folders = sorted([str(f.parts[-1]) for f in z_path.iterdir() if f.is_dir()])
    name = folders[args.job_id]
    # analysis_key = "{}/{}/".format(args.project, name)
else:
    name = args.name

# Read in config file with all parameters and settings
config = read.config(
    "{}/{}/{}/model_config.yaml".format(args.out_path, args.project, name)
)

# Initialize Weights & Biases
run = wandb.init(
    project=args.project, name=name, config=config, dir=args.out_path + args.project + "/" + name
)
print("WANDB directory: {}".format(run.dir))

# Get DataLoaders and model
loader_dict, model = befound.get.data_and_model(
    config,
    train_val_test=["train","val"],
    data_keys=["x6d", "root", "offsets", "target_pose"],
    shuffle=[True,False],
    use_default_offsets=[True,False],
)

# Train model
model = befound.train.train(config, model, loader_dict, run)

run.finish()
