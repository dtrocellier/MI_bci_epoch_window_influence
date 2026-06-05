#!/bin/bash
#SBATCH --job-name=mi_subjects
#SBATCH --array=1-87
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err

set -euo pipefail

SUBJECT=${SLURM_ARRAY_TASK_ID}

apptainer exec --nv \
  eeg-set-transformer.sif \
  python main.py \
  "$@" \
  subject=${SLURM_ARRAY_TASK_ID}