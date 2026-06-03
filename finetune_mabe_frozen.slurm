#!/bin/bash
#SBATCH --gres=gpu:a5000:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --job-name=ft_frozen
#SBATCH --output=logs/%A_%a.out
#SBATCH --error=logs/%A_%a.err
#SBATCH --account=tdunn
#SBATCH --mem=80G
#SBATCH --partition=tdunn-gpu

source /hpc/group/tdunn/yw789/miniconda3/etc/profile.d/conda.sh
conda activate befound

/hpc/group/tdunn/yw789/miniconda3/envs/befound/bin/python finetune_mabe22_frozen.py \
  -p ci_vae/finetune -n mabe22_ft_frozen \
  --out_path /hpc/group/tdunn/yw789/befound_results/
