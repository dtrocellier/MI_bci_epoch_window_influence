#!/bin/bash
#SBATCH --job-name=mi_epoch_window_influence
#SBATCH --time=00:10:00
#SBATCH --constraint="sirocco&p100"
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH -o /home/%u/output_job/test/job-%x.%N.%j.out
#SBATCH -e /home/%u/output_job/test/job-%x.%N.%j.err

/usr/bin/apptainer exec --nv \
  /beegfs/skojima/docker/mi-bci-epoch-window-influence.sif \
  python -c 'print("Hello World")'
