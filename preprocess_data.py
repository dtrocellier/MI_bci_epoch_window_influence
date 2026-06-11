import importlib
import json
import os
from pathlib import Path

import mne
import numpy as np
import yaml

from src.utils import load_config, window_suffix

mne.set_log_level(verbose="CRITICAL")


def preprocess_raw(raw):
    config = load_config()
    FILTER = config["filter"]
    raw.load_data()
    raw.pick(picks="eeg")
    raw.filter(l_freq=FILTER[0], h_freq=FILTER[1])
    return raw


def _split_data(X, y, block_size):
    I_0 = np.where(y == 0)[0]
    I_1 = np.where(y == 1)[0]

    assert (
        I_0.shape[0] == I_1.shape[0]
    ), "Number of trials for class 0 and class 1 should be equal"
    assert (
        I_0.shape[0] % block_size == 0
    ), f"Number of trials for class 0 and class 1 should be divisible by {block_size}"

    I_0 = I_0.reshape(-1, block_size)
    I_1 = I_1.reshape(-1, block_size)

    X_list, y_list = [], []

    for i_0, i_1 in zip(I_0, I_1):
        I = np.concatenate([i_0, i_1])
        I = np.sort(I)
        X_list.append(X[I])
        y_list.append(y[I])

    return X_list, y_list


def _save_data(X, y, save_base, suffix, dataset_name, subject, session_num, run_num):
    if dataset_name == "Lee2019_MI":

        X_list, y_list = _split_data(X, y, 5)

        for run_idx, (X, y) in enumerate(zip(X_list, y_list)):
            np.save(
                save_base
                / f"sub-{subject}_ses-{session_num}_run-{run_idx + 1}_X_{suffix}.npy",
                X,
            )

            np.save(
                save_base
                / f"sub-{subject}_ses-{session_num}_run-{run_idx + 1}_y_{suffix}.npy",
                y,
            )

    else:
        np.save(
            save_base / f"sub-{subject}_ses-{session_num}_run-{run_num}_X_{suffix}.npy",
            X,
        )

        np.save(
            save_base / f"sub-{subject}_ses-{session_num}_run-{run_num}_y_{suffix}.npy",
            y,
        )


def export_epochs(dataset, dataset_name, tmin, tmax, sfreq, save_base, suffix):
    config = load_config()
    for subject in dataset.subject_list:
        data = dataset.get_data(
            subjects=[subject], cache_config=config["MOABB"]["cache_config"]
        )[subject]
        for session_idx, (session_name, session_data) in enumerate(data.items()):
            for run_idx, (run_name, run_raw) in enumerate(session_data.items()):
                print(
                    f"Exporting data for subject {subject}, session {session_idx + 1}, run {run_idx + 1}"
                )
                run_raw = preprocess_raw(run_raw)

                events, event_id = mne.events_from_annotations(
                    run_raw, config["event_id"]
                )

                epochs = mne.Epochs(
                    run_raw,
                    events=events,
                    event_id=event_id,
                    tmin=-1,
                    tmax=5,
                    baseline=(None, 0),
                )
                epochs.load_data()

                epochs = epochs.crop(tmin=tmin, tmax=tmax)

                if sfreq is not None:
                    epochs = epochs.resample(sfreq)

                X = epochs.get_data()
                y = epochs.events[:, 2]

                _save_data(
                    X=X,
                    y=y,
                    save_base=save_base,
                    suffix=suffix,
                    dataset_name=dataset_name,
                    subject=subject,
                    session_num=session_idx + 1,
                    run_num=run_idx + 1,
                )


def export_meta_data(dataset, save_base, subject=1):
    data = dataset.get_data(subjects=[subject], cache_config={"use": True})[subject]
    for session_name, session_data in data.items():
        for run_name, run_raw in session_data.items():
            run_raw = preprocess_raw(run_raw)
            ch_names = run_raw.ch_names
            sfreq = run_raw.info["sfreq"]

            meta_data = {
                "ch_names": ch_names,
                "sfreq": sfreq,
            }

            with open(save_base / f"meta_data.json", "w") as f:
                json.dump(meta_data, f)

            break
        break


def load_yaml(files, base):
    yaml_list = []
    for file in files:
        if file.endswith(".yaml"):
            with open(base / file, "r") as f:
                yaml_list.append(yaml.safe_load(f))
    return yaml_list


def parse_config():
    conf_base = Path("conf")

    # dataset_list
    datasets = os.listdir(conf_base / "dataset")
    datasets = [dataset.split(".")[0] for dataset in datasets]

    # sfreq_list
    files = os.listdir(conf_base / "model")
    models = load_yaml(files, conf_base / "model")
    sfreq_list = [model["sfreq"] for model in models if model["name"] != "CSP_LDA"]
    sfreq_list = list(set(sfreq_list))

    # epoch_windows
    files = os.listdir(conf_base / "dataset")
    ds_yaml_list = load_yaml(files, conf_base / "dataset")
    epoch_windows = {
        f"{ds_yaml['name']}": ds_yaml["epoch_windows"] for ds_yaml in ds_yaml_list
    }

    return datasets, sfreq_list, epoch_windows


if __name__ == "__main__":

    DATASETS, SFREQ, EPOCH_WINDOWS = parse_config()

    DATASETS = ["Dreyer2023", "Lee2019_MI"]

    # DATASETS = ["Lee2019_MI"]
    # SFREQ = [200]
    print(SFREQ)

    for dataset_name in DATASETS:
        save_base = Path("Dataset") / dataset_name
        save_base.mkdir(exist_ok=True, parents=True)

        module = importlib.import_module("moabb.datasets")
        dataset = getattr(module, dataset_name)()

        export_meta_data(dataset, save_base)

        for sfreq in SFREQ:
            for tmin, tmax in EPOCH_WINDOWS[dataset_name]:
                suffix = window_suffix(tmin, tmax, sfreq)
                print(
                    f"\n=== {dataset_name} Epoch window: tmin={tmin}s  tmax={tmax}s  (suffix: {suffix}) ==="
                )

                export_epochs(
                    dataset=dataset,
                    dataset_name=dataset_name,
                    tmin=tmin,
                    tmax=tmax,
                    sfreq=sfreq,
                    save_base=save_base,
                    suffix=suffix,
                )

    print("\nDone preprocessing all windows.")
