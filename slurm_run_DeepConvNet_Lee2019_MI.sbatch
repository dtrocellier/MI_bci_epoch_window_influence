#!/bin/bash
#SBATCH --job-name=DCN_Lee
##SBATCH --constraint="sirocco&a100&zen3"
#SBATCH --constraint="sirocco&rtx8000"
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --exclusive
#SBATCH --time=05:00:00
#SBATCH -o /home/%u/output_job/mi_epoch_window_influence/job-%x.%N.%j.out
#SBATCH -e /home/%u/output_job/mi_epoch_window_influence/job-%x.%N.%j.err

mkdir -p logs

CUDA_VISIBLE_DEVICES=0 uv run python -u main.py -m \
    dataset=Lee2019_MI \
    epoch_window=w_00_30 \
    model=Deep4Net \
    subject=$(seq -s, 1 54) \
    > logs/lee_w_00_30_${SLURM_JOB_ID}.out \
    2> logs/lee_w_00_30_${SLURM_JOB_ID}.err &

CUDA_VISIBLE_DEVICES=1 uv run python -u main.py -m \
    dataset=Lee2019_MI \
    epoch_window=w_05_35 \
    model=Deep4Net \
    subject=$(seq -s, 1 54) \
    > logs/lee_w_05_35_${SLURM_JOB_ID}.out \
    2> logs/lee_w_05_35_${SLURM_JOB_ID}.err &

wait