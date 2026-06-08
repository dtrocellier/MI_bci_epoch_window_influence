#!/bin/bash
#SBATCH -o /home/%u/output_job/mi_epoch_window_influence/job-%x.%N.%j.out
#SBATCH -e /home/%u/output_job/mi_epoch_window_influence/job-%x.%N.%j.err

SUBJECTS=($(python get_subject_list.py dataset=Dreyer2023))
export SUBJECT_LIST="${SUBJECTS[*]}"

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

sbatch --array=0-$((${#SUBJECTS[@]} - 1)) run.sbatch \
    dataset=Dreyer2023 \
    model=Deep4Net