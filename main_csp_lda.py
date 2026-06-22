from pathlib import Path
import numpy as np
from pprint import pprint

import hydra
from omegaconf import OmegaConf

from sklearn.model_selection import train_test_split
import mne
from sklearn.pipeline import Pipeline
from mne.decoding import CSP
from sklearn.metrics import accuracy_score

from src.utils import SmartSave, load_test_runs
from src.classifier import LDA

mne.set_log_level(verbose="CRITICAL")


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
    print(f"window: {cfg.epoch_window}")
    print(f"subject: {cfg.subject}")

    sfreq = cfg.model.sfreq
    tmin = cfg.epoch_window.tmin
    tmax = cfg.epoch_window.tmax

    suffix = f"{tmin}_{tmax}_{sfreq}"

    smart_save = SmartSave(
        Path("results") / "classification_results_csp_lda.csv",
        columns=[
            "mode",
            "dataset",
            "subject",
            "session",
            "model",
            "epoch_window",
            "accuracy",
        ],
    )

    base = Path("Dataset") / cfg.dataset.name / "CSP_LDA"

    for session_idx in range(cfg.dataset.n_sessions):
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

        print(f"train_X: {train_X.shape}, test_X: {test_X.shape}")
        print(f"train_y: {train_y.shape}, test_y: {test_y.shape}")

        model = Pipeline(
            [
                ("CSP", CSP(n_components=cfg.model.n_components)),
                ("LDA", LDA()),
            ]
        )

        model.fit(train_X, train_y)

        preds = model.predict(test_X)

        acc = accuracy_score(test_y, preds)

        smart_save(
            dict(
                mode="within_user",
                dataset=cfg.dataset.name,
                subject=cfg.subject,
                session=session_idx + 1,
                model="CSP_LDA",
                epoch_window=cfg.epoch_window.name,
            ),
            dict(accuracy=acc),
        )

        print(
            f"Done (within-user): {cfg.model.name} | {cfg.dataset.name} | {cfg.epoch_window.name} | subject {cfg.subject} | session {session_idx + 1} → {acc:.4f}"
        )


def run_cross_user(cfg):
    print(f"window: {cfg.epoch_window}")
    print(f"subject: {cfg.subject}")

    sfreq = cfg.model.sfreq
    tmin = cfg.epoch_window.tmin
    tmax = cfg.epoch_window.tmax

    suffix = f"{tmin}_{tmax}_{sfreq}"

    smart_save = SmartSave(
        Path("results") / "classification_results_csp_lda.csv",
        columns=[
            "mode",
            "dataset",
            "subject",
            "session",
            "model",
            "epoch_window",
            "accuracy",
        ],
    )

    base = Path("Dataset") / cfg.dataset.name / "CSP_LDA"

    subjects = list(range(1, cfg.dataset.subjects.n_subjects + 1))
    if cfg.dataset.subjects.exclude is not None:
        for subject in cfg.dataset.subjects.exclude:
            subjects.remove(subject)

    # remove test subject
    test_subject = cfg.subject
    subjects.remove(test_subject)
    train_subjects = subjects.copy()

    if cfg.debug:
        train_subjects = train_subjects[:10]

    print("train_subjects: ", train_subjects)
    print("test_subject: ", test_subject)

    # load training data
    X_list, y_list = [], []
    for subject in train_subjects:
        train_runs = cfg.dataset.runs.train
        test_runs = cfg.dataset.runs.test

        for session_idx in range(cfg.dataset.n_sessions):
            X, y = load_runs(
                runs=train_runs + test_runs,
                base=base,
                session_idx=session_idx,
                suffix=suffix,
                subject=subject,
            )
        X_list.append(X)
        y_list.append(y)
    train_X = np.concatenate(X_list)
    train_y = np.concatenate(y_list)

    print(f"train_X: {train_X.shape}, train_y: {train_y.shape}")

    model = Pipeline(
        [
            ("CSP", CSP(n_components=cfg.model.n_components)),
            ("LDA", LDA()),
        ]
    )

    model.fit(train_X, train_y)

    for session_idx in range(cfg.dataset.n_sessions):
        test_runs = cfg.dataset.runs.test

        test_X, test_y = load_runs(
            runs=test_runs,
            base=base,
            session_idx=session_idx,
            suffix=suffix,
            subject=test_subject,
        )

        print(f"test_X: {test_X.shape}, test_y: {test_y.shape}")

        preds = model.predict(test_X)

        acc = accuracy_score(test_y, preds)

        smart_save(
            dict(
                mode="cross_user",
                dataset=cfg.dataset.name,
                subject=cfg.subject,
                session=session_idx + 1,
                model="CSP_LDA",
                epoch_window=cfg.epoch_window.name,
            ),
            dict(accuracy=acc),
        )
        print(
            f"Done (cross-user): {cfg.model.name} | {cfg.dataset.name} | {cfg.epoch_window.name} | subject {cfg.subject} | session {session_idx + 1} → {acc:.4f}"
        )


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg):
    if cfg.debug:
        from src.utils import debug_warning

        debug_warning()

    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    pprint(cfg_dict)
    if cfg.dataset.subjects.exclude is not None:
        if cfg.subject in cfg.dataset.subjects.exclude:
            print(f"subject: {cfg.subject} is excluded.")
            return

    if cfg.model.name != "CSP_LDA":
        raise ValueError("model name must be CSP_LDA")

    run_within_user(cfg)
    run_cross_user(cfg)


if __name__ == "__main__":
    main()
