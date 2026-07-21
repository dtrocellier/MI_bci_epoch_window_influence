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


def export_epochs(dataset, dataset_name, save_base):
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
                    tmin=-3.5,
                    tmax=5.5,
                    baseline=None,
                )

                if dataset_name == "Lee2019_MI":
                    epochs.load_data()
                    epochs.resample(512)

                epochs.save(
                    save_base
                    / f"sub-{subject}_ses-{session_idx + 1}_run-{run_idx + 1}-epo.fif",
                    overwrite=True,
                )


def load_yaml(files, base):
    yaml_list = []
    for file in files:
        if file.endswith(".yaml"):
            with open(base / file, "r") as f:
                yaml_list.append(yaml.safe_load(f))
    return yaml_list


if __name__ == "__main__":

    # DATASETS = ["Dreyer2023", "Lee2019_MI"]
    # DATASETS = ["Lee2019_MI"]
    DATASETS = ["Dreyer2023"]

    for dataset_name in DATASETS:
        save_base = Path("Dataset") / "epochs" / dataset_name
        save_base.mkdir(exist_ok=True, parents=True)

        module = importlib.import_module("moabb.datasets")
        dataset = getattr(module, dataset_name)()

        # export_meta_data(dataset, save_base)

        export_epochs(
            dataset=dataset,
            dataset_name=dataset_name,
            save_base=save_base,
        )

    print("\nDone preprocessing all windows.")
