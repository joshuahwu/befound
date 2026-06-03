#!/bin/bash
#SBATCH --gres=gpu:2
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=4
#SBATCH --job-name=train_bespoke
#SBATCH --output=logs/%A_%a.out
#SBATCH --error=logs/%A_%a.err
#SBATCH --account=tdunn
#SBATCH --mem=150G
#SBATCH --partition=tdunn-gpu

source /hpc/group/tdunn/yw789/miniconda3/etc/profile.d/conda.sh
conda activate befound

torchrun \
  --nproc_per_node=$SLURM_NTASKS_PER_NODE \
  --nnodes=$SLURM_NNODES \
  --node_rank=$SLURM_NODEID \
  --master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1) \
  --master_port=29500 \
train_model_2d_bespoke_ddp.py -p f_vae -n 2d_bespoke_mabe
