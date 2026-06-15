#!/bin/bash
#SBATCH --job-name=REVE_Dreyer
#SBATCH --constraint="sirocco&rtx8000"
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=40
#SBATCH --exclusive
#SBATCH --time=2-00:00:00
#SBATCH -o /home/%u/output_job/mi_epoch_window_influence/job-%x.%N.%j.out
#SBATCH -e /home/%u/output_job/mi_epoch_window_influence/job-%x.%N.%j.err

export OMP_NUM_THREADS=20
export MKL_NUM_THREADS=20
export OPENBLAS_NUM_THREADS=20
export NUMEXPR_NUM_THREADS=20
export TORCH_NUM_THREADS=20

mkdir -p logs

OMP_NUM_THREADS=20 MKL_NUM_THREADS=20 OPENBLAS_NUM_THREADS=20 \
CUDA_VISIBLE_DEVICES=0 taskset -c 0-19 \
uv run python -u main.py -m \
    dataset=Dreyer2023 \
    epoch_window=w_00_40 \
    model=REVE \
    subject=$(seq -s, 49 87) \
    > logs/REVE_dreyer_w_00_40_${SLURM_JOB_ID}.out \
    2> logs/REVE_dreyer_w_00_40_${SLURM_JOB_ID}.err &

OMP_NUM_THREADS=20 MKL_NUM_THREADS=20 OPENBLAS_NUM_THREADS=20 \
CUDA_VISIBLE_DEVICES=1 taskset -c 20-39 \
uv run python -u main.py -m \
    dataset=Dreyer2023 \
    epoch_window=w_05_45 \
    model=REVE \
    subject=$(seq -s, 56 87) \
    > logs/REVE_dreyer_w_05_45_${SLURM_JOB_ID}.out \
    2> logs/REVE_dreyer_w_05_45_${SLURM_JOB_ID}.err &

wait
