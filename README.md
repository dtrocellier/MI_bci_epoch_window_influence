# Epoch Window Influence on EEG Motor Imagery Classification

![Python](https://img.shields.io/badge/Python-3.11-blue) ![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange) ![MNE](https://img.shields.io/badge/MNE-EEG-green)

## Overview

This project investigates how the **choice of epoch window** affects EEG-based motor imagery (MI) classification. It
compares two windows extracted from the same raw EEG recordings: one starting at visual cue onset and the other starting
0.5 seconds later.

The experiments use the Dreyer2023 and Lee2019_MI datasets. The windows used for each dataset are shown below.

### Dreyer2023 dataset

| Window | Start (tmin) | End (tmax) | Duration |
|--------|--------------|------------|----------|
| A      | 0.0 s        | 4.0 s      | 4 s      |
| B      | 0.5 s        | 4.5 s      | 4 s      |

### Lee2019_MI dataset

| Window | Start (tmin) | End (tmax) | Duration |
|--------|--------------|------------|----------|
| A      | 0.0 s        | 3.0 s      | 3 s      |
| B      | 0.5 s        | 3.5 s      | 3 s      |

The two windows have the same duration within each dataset but start at different times relative to the MI cue. The
experiments assess whether this shift changes classification performance and cross-subject generalization.

Models are evaluated using **leave-one-subject-out cross-validation**.

---

## How It Works

```text
Dataset (downloaded through MOABB)
                │
                ▼
preprocess_data.py / preprocess_data_csp_lda.py
(epoch windows configured in conf/)
                │
                ▼
Dataset/<dataset>/
(preprocessed NumPy files)
                │
                ▼
main.py / main_csp_lda.py
(experiments configured with Hydra)
                │
                ▼
results/ | attributions/ | models/
```

---

## Repository Layout

```text
MI_bci_epoch_window_influence/
├── conf/                       # Hydra configuration files
│   ├── dataset/                # Dataset configurations
│   ├── epoch_window/           # Epoch window definitions
│   ├── model/                  # Model configurations
│   └── config.yaml             # Base configuration
├── preprocess_data.py          # Preprocess data for each epoch window
├── preprocess_data_csp_lda.py  # Preprocess data for CSP + LDA
├── main.py                     # Run deep learning experiments
├── main_csp_lda.py             # Run CSP + LDA experiments
├── notebooks/                  # Analysis and visualization
├── Dataset/                    # Raw and preprocessed data (gitignored)
├── models/                     # Saved checkpoints (gitignored)
└── results/                    # Classification results (gitignored)
```

---

## Prerequisites

- Python 3.11
- [uv](https://docs.astral.sh/uv/)

Install the dependencies with:

```bash
uv sync
```

---

## Step-by-Step Execution

### Step 1 — Preprocess the data

```bash
uv run python preprocess_data.py
uv run python preprocess_data_csp_lda.py
```

For each window, the preprocessed data are saved under `Dataset/<dataset>/` as files such as:

```text
sub-<subject>_ses-<session>_run-<run>_X_<window>_<sampling_frequency>.npy
sub-<subject>_ses-<session>_run-<run>_y_<window>_<sampling_frequency>.npy
```

### Step 2 — Run the experiments

The commands below use Window A and Deep4Net as examples. To run Window B, replace `w_00_40` with `w_05_45` for
Dreyer2023, or `w_00_30` with `w_05_35` for Lee2019_MI. For the deep learning experiments, replace `Deep4Net` with
`REVE` to run the other model.

**Dreyer2023 — deep learning**

```bash
uv run python -u main.py -m \
    dataset=Dreyer2023 \
    epoch_window=w_00_40 \
    model=Deep4Net \
    subject=$(seq -s, 1 87)
```

**Lee2019_MI — deep learning**

```bash
uv run python -u main.py -m \
    dataset=Lee2019_MI \
    epoch_window=w_00_30 \
    model=Deep4Net \
    subject=$(seq -s, 1 54)
```

**Dreyer2023 without P-line electrodes — deep learning**

```bash
uv run python -u main.py -m \
    dataset=Dreyer2023_reject_P \
    epoch_window=w_00_40 \
    model=Deep4Net \
    subject=$(seq -s, 1 87)
```

**Dreyer2023 — CSP + LDA**

```bash
uv run python -u main_csp_lda.py -m \
    dataset=Dreyer2023 \
    epoch_window=w_00_40 \
    model=CSP_LDA \
    subject=$(seq -s, 1 87)
```

**Lee2019_MI — CSP + LDA**

```bash
uv run python -u main_csp_lda.py -m \
    dataset=Lee2019_MI \
    epoch_window=w_00_30 \
    model=CSP_LDA \
    subject=$(seq -s, 1 54)
```