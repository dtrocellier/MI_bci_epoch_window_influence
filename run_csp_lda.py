from pathlib import Path
import numpy as np
import importlib

import hydra
from omegaconf import OmegaConf

from sklearn.pipeline import Pipeline
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from mne.decoding import CSP
from sklearn.metrics import accuracy_score

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

    base = Path("Dataset") / cfg.dataset.name / "CSP_LDA"

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

        model = Pipeline(
            [
                ("CSP", CSP(n_components=cfg.model.n_components, log=True)),
                ("LDA", LinearDiscriminantAnalysis(solver="eigen", shrinkage="auto")),
            ]
        )

        model.fit(train_X, train_y)

        preds = model.predict(test_X)

        acc = accuracy_score(test_y, preds)

        print(acc)

        save_results(cfg, acc, "classification_results_csp_lda.csv")

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
