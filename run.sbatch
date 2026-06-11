#!/bin/bash
#SBATCH --job-name=mi_epoch_window_influence
#SBATCH --constraint="sirocco&p100"
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH -o /home/%u/output_job/mi_epoch_window_influence/job-%x.%N.%j.out
#SBATCH -e /home/%u/output_job/mi_epoch_window_influence/job-%x.%N.%j.err

export CUDA_VISIBLE_DEVICES=0


/usr/bin/apptainer exec --nv \
  /beegfs/skojima/docker/mi-bci-epoch-window-influence.sif \
  python -c "
import torch
print('torch:', torch.__version__)
print('cuda available:', torch.cuda.is_available())
print('cuda version:', torch.version.cuda)
print('device count:', torch.cuda.device_count())
if torch.cuda.is_available():
    print('device name:', torch.cuda.get_device_name(0))
"

/usr/bin/apptainer exec --nv \
  /beegfs/skojima/docker/mi-bci-epoch-window-influence.sif \
  python main.py \
  "$@" \
  subject=$SUBJECT