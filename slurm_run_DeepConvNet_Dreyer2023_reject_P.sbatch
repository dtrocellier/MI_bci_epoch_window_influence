#!/bin/bash
#SBATCH --job-name=DCN_Dreyer
##SBATCH --constraint="sirocco&a100&zen3"
#SBATCH --constraint="sirocco&rtx8000"
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --exclusive
#SBATCH --time=12:00:00
#SBATCH -o /home/%u/output_job/mi_epoch_window_influence/job-%x.%N.%j.out
#SBATCH -e /home/%u/output_job/mi_epoch_window_influence/job-%x.%N.%j.err

export OMP_NUM_THREADS=16
export MKL_NUM_THREADS=16
export OPENBLAS_NUM_THREADS=16
export NUMEXPR_NUM_THREADS=16
export TORCH_NUM_THREADS=16

mkdir -p logs

OMP_NUM_THREADS=16 MKL_NUM_THREADS=16 OPENBLAS_NUM_THREADS=16 \
CUDA_VISIBLE_DEVICES=0 taskset -c 0-15 \
uv run python -u main.py -m \
    dataset=Dreyer2023_reject_P \
    epoch_window=w_00_40 \
    model=Deep4Net \
    subject=$(seq -s, 1 87) \
    > logs/dreyer_w_00_40_${SLURM_JOB_ID}.out \
    2> logs/dreyer_w_00_40_${SLURM_JOB_ID}.err &


OMP_NUM_THREADS=16 MKL_NUM_THREADS=16 OPENBLAS_NUM_THREADS=16 \
CUDA_VISIBLE_DEVICES=1 taskset -c 16-31 \
uv run python -u main.py -m \
    dataset=Dreyer2023_reject_P \
    epoch_window=w_05_45 \
    model=Deep4Net \
    subject=$(seq -s, 1 87) \
    > logs/dreyer_w_05_45_${SLURM_JOB_ID}.out \
    2> logs/dreyer_w_05_45_${SLURM_JOB_ID}.err &

wait