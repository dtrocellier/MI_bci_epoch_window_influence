#!/bin/bash
#SBATCH --job-name=cross-deeplearning
#SBATCH --constraint="sirocco&a100&zen3"
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --exclusive
#SBATCH --time=05:00:00
#SBATCH -o /home/%u/output_job/mi_epoch_window_influence/job-%x.%N.%j.out
#SBATCH -e /home/%u/output_job/mi_epoch_window_influence/job-%x.%N.%j.err

mkdir -p logs

CUDA_VISIBLE_DEVICES=0 uv run python main.py -m \
    dataset=Dreyer2023 \
    epoch_window=w_00_40 \
    model=REVE \
    subject=$(seq -s, 1 87) \
    > logs/dreyer_w_00_40_${SLURM_JOB_ID}.out \
    2> logs/dreyer_w_00_40_${SLURM_JOB_ID}.err &

CUDA_VISIBLE_DEVICES=1 uv run python main.py -m \
    dataset=Dreyer2023 \
    epoch_window=w_05_45 \
    model=REVE \
    subject=$(seq -s, 1 87) \
    > logs/dreyer_w_05_45_${SLURM_JOB_ID}.out \
    2> logs/dreyer_w_05_45_${SLURM_JOB_ID}.err &
