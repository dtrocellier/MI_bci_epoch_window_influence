#!/bin/bash
#SBATCH --job-name=CSP_LDA_Lee
#SBATCH --constraint="suroit"
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=48
#SBATCH --exclusive
#SBATCH --time=12:00:00
#SBATCH -o /home/%u/output_job/mi_epoch_window_influence/job-%x.%N.%j.out
#SBATCH -e /home/%u/output_job/mi_epoch_window_influence/job-%x.%N.%j.err

export OMP_NUM_THREADS=24
export MKL_NUM_THREADS=24
export OPENBLAS_NUM_THREADS=24
export NUMEXPR_NUM_THREADS=24

mkdir -p logs

OMP_NUM_THREADS=24 MKL_NUM_THREADS=24 OPENBLAS_NUM_THREADS=24 NUMEXPR_NUM_THREADS=24 \
taskset -c 0-23 \
uv run python -u run_csp_lda.py -m \
    dataset=Lee2019_MI \
    epoch_window=w_00_30 \
    model=CSP_LDA \
    subject=$(seq -s, 1 87) \
    > logs/CSP_LDA_lee_w_00_40_${SLURM_JOB_ID}.out \
    2> logs/CSP_LDA_lee_w_00_40_${SLURM_JOB_ID}.err &


OMP_NUM_THREADS=24 MKL_NUM_THREADS=24 OPENBLAS_NUM_THREADS=24 NUMEXPR_NUM_THREADS=24 \
taskset -c 24-47 \
uv run python -u run_csp_lda.py -m \
    dataset=Lee2019_MI \
    epoch_window=w_05_35 \
    model=CSP_LDA \
    subject=$(seq -s, 1 54) \
    > logs/CSP_LDA_lee_w_05_45_${SLURM_JOB_ID}.out \
    2> logs/CSP_LDA_lee_w_05_45_${SLURM_JOB_ID}.err &

wait
