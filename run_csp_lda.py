from pathlib import Path
import numpy as np
import importlib

import hydra
from omegaconf import OmegaConf

from sklearn.pipeline import Pipeline
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from mne.decoding import CSP

from src.utils import (
    save_results,
)


def load_runs(runs, base, session_idx, suffix, subject):
    X_list, y_list = [], []
    for run in runs:
        fname = base / f"sub-{subject}_ses-{session_idx + 1}_run-{run}_X_{suffix}.npy"
        X = np.load(fname)

        fname = base / f"sub-{subject}_ses-{session_idx + 1}_run-{run}_y_{suffix}.npy"
        y = np.load(fname)

        X_list.append(X)
        y_list.append(y)
    X = np.concatenate(X_list)
    y = np.concatenate(y_list)
    return X, y


def run_within_user(cfg):
    print(cfg.dataset)

    print(f"window: {cfg.epoch_window}")
    print(f"subject: {cfg.subject}")

    sfreq = cfg.model.sfreq
    tmin = cfg.epoch_window.tmin
    tmax = cfg.epoch_window.tmax

    suffix = f"{tmin}_{tmax}_{sfreq}"

    module = importlib.import_module("moabb.datasets")
    cls = getattr(module, cfg.dataset.name)

    dataset = cls()

    cache_config = OmegaConf.to_container(cfg.MOABB.cache_config)
    print(cache_config)
    print(type(cache_config))

    data = dataset.get_data(subjects=[cfg.subject], cache_config=cache_config)[
        cfg.subject
    ]

    print(data)

    for ses_name, ses_data in data.items():
        print(ses_name)
        print(ses_data)
        for run_name, raw in ses_data.items():
            print(run_name)
            print(raw)
            break
        break

    exit()

    for session_idx in range(cfg.dataset.n_sessions):
        print(session_idx)

        train_runs = cfg.dataset.runs.train
        test_runs = cfg.dataset.runs.test

        train_X, train_y = load_runs(
            runs=train_runs,
            base=base,
            session_idx=session_idx,
            suffix=suffix,
            subject=cfg.subject,
        )

        test_X, test_y = load_runs(
            runs=test_runs,
            base=base,
            session_idx=session_idx,
            suffix=suffix,
            subject=cfg.subject,
        )

        print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)

        exit()


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg):
    if cfg.dataset.subjects.exclude is not None:
        if cfg.subject in cfg.dataset.subjects.exclude:
            print(f"subject: {cfg.subject} is excluded.")
            return

    if cfg.model.name != "CSP_LDA":
        raise ValueError("model name must be CSP_LDA")

    run_within_user(cfg)


if __name__ == "__main__":
    main()
