import sys
import os
import torch
import wandb

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from config import SWEEP_PARAMS, SWEEP_PROJECT, SWEEP_ID_FILE, DATA_PATH, EPOCH_WINDOWS, window_suffix

# Derive subject list from the preprocessed data (any window works — all have the same subjects)
first_suffix = window_suffix(*EPOCH_WINDOWS[0])
y_path = os.path.join(DATA_PATH, 'Large', f'Y_s_{first_suffix}.pt')
Y = torch.load(y_path)
all_subjects = list(range(len(Y)))
print(f"Found {len(all_subjects)} subjects in {y_path}")

sweep_params = dict(SWEEP_PARAMS)
sweep_params['test_subject'] = {'values': all_subjects}

sweep_config = {'method': 'grid', 'parameters': sweep_params}

sweep_id = wandb.sweep(sweep_config, project=SWEEP_PROJECT)

os.makedirs(os.path.dirname(SWEEP_ID_FILE), exist_ok=True)
with open(SWEEP_ID_FILE, 'w') as f:
    f.write(sweep_id)

print(f"Sweep created: {sweep_id}")
print(f"Grid: {len(all_subjects)} subjects × {len(EPOCH_WINDOWS)} windows = {len(all_subjects) * len(EPOCH_WINDOWS)} runs")
print(f"Sweep ID saved to: {SWEEP_ID_FILE}")