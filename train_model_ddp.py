import os
import befound
from befound.params import read
from pathlib import Path
import wandb
import argparse
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler


def setup_ddp():
    """Initialize the distributed process group using env vars set by torchrun."""
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank, dist.get_rank(), dist.get_world_size()


def cleanup_ddp():
    """Destroy the distributed process group."""
    dist.destroy_process_group()


def main():
    local_rank, rank, world_size = setup_ddp()

    # argparse project and job names
    parser = argparse.ArgumentParser(prog="Behavioral Foundation Models Train", description="Train Behavioral Foundation Models")
    parser.add_argument("--out_path", "-o", type=str, dest="out_path")
    parser.add_argument("--job_id", type=int, dest="job_id")
    parser.add_argument("--project", "-p", type=str, dest="project")
    parser.add_argument("--name", "-n", type=str, dest="name")
    args = parser.parse_args()

    if args.out_path is None:
        args.out_path = "/hpc/group/tdunn/yw789/befound_results"

    ### Set/Load Parameters
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

    # Initialize Weights & Biases on rank 0 only
    if rank == 0:
        wandb.login()
        run = wandb.init(
            project=args.project, name=name, config=config,
            dir=args.out_path + args.project + "/" + name
        )
        print("WANDB directory: {}".format(run.dir))
    else:
        run = None

    # Get DataLoaders and model
    # shuffle=False because DistributedSampler handles shuffling
    loader_dict, model = befound.get.data_and_model(
        config,
        train_val_test=["train", "val"],
        data_keys=["x6d", "root", "offsets", "target_pose"],
        shuffle=[False, False],
        use_default_offsets=[True, False],
    )

    # Replace each DataLoader with a DistributedSampler-based one
    for split, loader in loader_dict.items():
        sampler = DistributedSampler(
            loader.dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=(split == "train"),
        )
        loader_dict[split] = DataLoader(
            dataset=loader.dataset,
            batch_size=loader.batch_size,
            sampler=sampler,
            num_workers=loader.num_workers,
            pin_memory=loader.pin_memory,
        )

    # Wrap model with DDP
    # find_unused_parameters=True is required because the VAE forward() has
    # conditional branches (gaussian/beta/None prior) so not all parameters
    # receive gradients on every iteration.
    model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)

    # Train model (wandb run only on rank 0)
    model = befound.train.train(config, model, loader_dict, run, rank=rank)

    if rank == 0:
        run.finish()

    cleanup_ddp()


if __name__ == "__main__":
    main()

# Launch with torchrun:
# torchrun --nproc_per_node=NUM_GPUS train_model_ddp.py -p ci_vae -n 1_ablatel











