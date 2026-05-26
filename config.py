# Experiment dimensions — NOT W&B sweep parameters
MODELS        = ['Deep4Net']           # extend here to add models
DATASETS      = ['Dreyer2023', 'Lee2019_MI']             # extend here to add datasets
EPOCH_WINDOWS = [(0, 4), (0.5, 4.5)]  # (tmin, tmax) pairs


def window_suffix(tmin, tmax):
    return f"{tmin}_{tmax}"


# W&B sweep parameters — test_subject is added dynamically in bin/200_create_sweep_training.py
# by reading the preprocessed data to enumerate all available subjects
SWEEP_PARAMS = {
    'epoch_window': {'values': [window_suffix(t0, t1) for t0, t1 in EPOCH_WINDOWS]},
    'optimizer':    {'value': 'adamW'},
    'scheduler':    {'value': 'CosineAnnealingLR'},
    'lr':           {'value': 0.001},
    'weight_decay': {'value': 0.0005},
    'n_epochs':     {'value': 150},
    'batch_size':   {'value': 256},
    'lr_gamma':     {'value': 0.1},
    'lr_step_size': {'value': 0},
}

# Shared paths
DATA_PATH      = 'Dataset/'
SAVE_PERF_PATH = 'Results/perf_benchmark.csv'
SWEEP_ID_FILE  = 'bin/sweep_id/sweep_id_window_influence.txt'
SWEEP_PROJECT  = 'Leave_one_out_window_influence'
