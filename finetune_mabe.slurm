#!/bin/bash
#SBATCH --gres=gpu:a5000:2
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=4
#SBATCH --job-name=finetune_mabe
#SBATCH --output=logs/%A_%a.out
#SBATCH --error=logs/%A_%a.err
#SBATCH --account=tdunn
#SBATCH --mem=150G
#SBATCH --partition=tdunn-gpu

source /hpc/group/tdunn/yw789/miniconda3/etc/profile.d/conda.sh
conda activate befound

# derive a unique port from job ID to avoid conflicts when multiple jobs share a node
MASTER_PORT=$(( 20000 + SLURM_JOB_ID % 10000 ))

torchrun \
  --nproc_per_node=$SLURM_NTASKS_PER_NODE \
  --nnodes=$SLURM_NNODES \
  --node_rank=$SLURM_NODEID \
  --master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1) \
  --master_port=$MASTER_PORT \
  finetune_mabe22_ddp.py -p ci_vae/finetune -n mabe22_ft \
  --out_path /hpc/group/tdunn/yw789/befound_results/
