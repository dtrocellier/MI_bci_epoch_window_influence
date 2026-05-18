# Epoch Window Influence on EEG Motor Imagery Classification

![Python](https://img.shields.io/badge/Python-3.8%2B-blue) ![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange) ![MNE](https://img.shields.io/badge/MNE-EEG-green) ![W&B](https://img.shields.io/badge/Weights_%26_Biases-tracking-yellow)

## Overview

This project investigates how the **choice of epoch time window** affects the performance of deep learning models for **EEG-based motor imagery (MI) classification**. Specifically, it compares two epoch windows extracted from the same raw EEG recordings:

| Window | Start (tmin) | End (tmax) | Duration |
|--------|-------------|------------|----------|
| Window A | 0.0 s | 4.0 s | 4 s |
| Window B | 0.5 s | 4.5 s | 4 s |

Both windows have identical duration (4 seconds / 2048 samples at 512 Hz), but differ in their temporal alignment relative to the motor imagery cue. The experiment evaluates whether this subtle temporal shift captures meaningfully different neural patterns — and whether it impacts cross-subject generalization.

The evaluation protocol is **leave-one-out cross-validation** across all subjects in the dataset. All experiments are tracked with [Weights & Biases](https://wandb.ai).

---

## How It Works

```
Raw GDF files (Big_dataset, 512 Hz, ~32 channels)
                       │
                       ▼
           preprocess_data.py
   (all windows defined in config.py)
                       │
                       ▼
           Dataset/Large/
   X_s_0_4.pt, Y_s_0_4.pt
   X_s_0.5_4.5.pt, Y_s_0.5_4.5.pt ...
                       │
                       ▼
        bin/create_sweep.py
   (configure W&B grid-search sweep)
                       │
                       ▼
   sweep_id_window_influence.txt
                       │
                       ▼
          train_model.py
   (train Deep4Net, leave-one-out)
                       │
                       ▼
   Results/perf_benchmark.csv
   model/Deep4Net/*.pt
   W&B: Leave_one_out_window_influence
```

### Key steps

1. **Preprocessing** — Raw GDF files are loaded with MNE, EOG/EMG channels are dropped, and a 0.5–40 Hz bandpass filter is applied. Epochs are extracted for **all windows defined in `config.py`** in a single run, producing one set of tensors per window.

2. **Sweep creation** — A W&B grid-search sweep is configured from `config.py` (epoch windows, hyperparameters). Subject list is enumerated dynamically from the preprocessed data.

3. **Training** — Each sweep run loads the tensors for the configured window, trains on all subjects except the held-out test subject, and evaluates on that subject. Metrics are logged per epoch to W&B.

---

## Repository Layout

```
epoch_window_influence/
├── config.py                   # Single source of truth — windows, models, paths
├── preprocess_data.py          # Step 1 — preprocess all epoch windows
├── train_model.py              # Step 3 — train with W&B sweep
├── bin/
│   ├── create_sweep.py         # Step 2 — register W&B sweep
│   └── sweep_id/
│       └── sweep_id_window_influence.txt
├── src/                        # Shared library (imported by pipeline scripts)
│   ├── read_data.py            # Data loading and cross-subject split logic
│   └── utils.py                # Preprocessing helpers (preprocess, Epoching)
├── notebooks/                  # Analysis and visualization (run after training)
│   ├── 500_analysis_performances.ipynb
│   ├── 501–502_saliency_map*.ipynb
│   ├── 503–505_plots*.ipynb
│   ├── 600–601_csp_filters*.ipynb
│   └── 900–902_gif_erd_ers*.ipynb
├── Dataset/                    # Raw and preprocessed data (gitignored)
├── model/                      # Saved checkpoints (gitignored)
├── Results/                    # Output plots and CSVs (gitignored)
└── requirements.txt
```

---

## Prerequisites

**Python:** 3.8 or higher

**Install dependencies:**

```bash
pip install -r requirements.txt
```

**Authenticate with Weights & Biases:**

```bash
wandb login
```

**Dataset:** Place the raw GDF files under the following structure (path configured inside preprocessing scripts):

```
Dataset/
└── Big_dataset/
    └── <participant_id>/
        ├── <participant_id>_R1_acquisition.gdf
        ├── <participant_id>_R2_acquisition.gdf
        ├── <participant_id>_R3_onlineT.gdf
        ├── <participant_id>_R4_onlineT.gdf
        ├── <participant_id>_R5_onlineT.gdf
        └── <participant_id>_R6_onlineT.gdf
```

---

## Step-by-Step Execution

### Step 1 — Preprocess

```bash
python preprocess_data.py
```

Reads all raw GDF files, applies a 0.5–40 Hz bandpass filter, drops EOG/EMG channels, and epochs the signal for **every window defined in `config.py`** in a single run.

**Output:** `Dataset/Big_dataset/X_s_<window>.pt`, `Y_s_<window>.pt` for each window.

---

### Step 2 — Create the W&B Sweep

```bash
python bin/create_sweep.py
```

Registers a grid-search sweep in the W&B project `Leave_one_out_window_influence`. Subject list is enumerated automatically from the preprocessed data.

**Output:** `bin/sweep_id/sweep_id_window_influence.txt`

**Sweep configuration (set in `config.py`):**

| Parameter | Value |
|-----------|-------|
| Epoch windows | all windows in `EPOCH_WINDOWS` |
| Test subjects | all subjects (leave-one-out) |
| Optimizer | AdamW |
| Learning rate | 0.001 |
| Weight decay | 0.0005 |
| Batch size | 256 |
| Epochs | 150 |
| Scheduler | CosineAnnealingLR |

---

### Step 3 — Train

```bash
python train_model.py
```

Launches W&B agents that iterate over all (subject × window) combinations. Each run:

1. Loads preprocessed tensors for the configured epoch window
2. Splits data: train / validation (80/20 of non-test subjects), test (held-out subject, online sessions)
3. Builds Deep4Net and trains for the configured number of epochs
4. Saves the best checkpoint to `model/Deep4Net/`
5. Appends accuracy to `Results/perf_benchmark.csv`
6. Logs all metrics live to W&B

---

## Key Parameters

### EEG Signal

| Parameter | Value |
|-----------|-------|
| Sampling rate | 512 Hz |
| EEG channels | 27 (after removing EOG/EMG) |
| Classes | 2 (motor imagery left / right) |
| Bandpass filter | 0.5 – 40 Hz |

### Epoch Windows

| | Window A | Window B |
|---|----------|----------|
| tmin | 0.0 s | 0.5 s |
| tmax | 4.0 s | 4.5 s |
| Duration | 4 s | 4 s |
| Samples | 2048 | 2048 |
| Overlap | 1 s | 1 s |

### Training

| Parameter | Value |
|-----------|-------|
| Model (primary) | Deep4Net |
| Optimizer | AdamW |
| Learning rate | 0.001 |
| Weight decay | 0.0005 |
| Batch size | 256 |
| Epochs | 150 |
| LR scheduler | CosineAnnealingLR |
| Loss | NLLLoss |
| Random seed | 2002012 |

---

## Supported Model Architectures


The active model is set via `MODELS` in `config.py`.

| Model | Source | Description |
|-------|--------|-------------|
| **Deep4Net** | braindecode | 4-layer deep convolutional network; only model used in training |
---

## Outputs

| Path | Description |
|------|-------------|
| `Dataset/Big_dataset/X_s_<window>.pt` | Preprocessed EEG tensors per epoch window |
| `bin/sweep_id/sweep_id_window_influence.txt` | W&B sweep ID |
| `model/Deep4Net/<model>_<dataset>_<window>_<subject>_*.pt` | Best checkpoint per run |
| `Results/perf_benchmark.csv` | Accuracy per subject / window / model |
| `Results/*.png` | Visualization plots |
| `Results/*.npy` | Saliency maps (spatial and temporal) |

---

## Monitoring Results

All training runs are logged to the W&B project **`Leave_one_out_window_influence`**.

```bash
# Open the project dashboard
wandb project Leave_one_out_window_influence
```

Or visit [https://wandb.ai](https://wandb.ai) and navigate to your project to compare runs across epoch windows, subjects, and model architectures.
