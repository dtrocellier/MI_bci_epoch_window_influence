from pathlib import Path
import importlib
import mne
import os
import natsort
from joblib import Parallel, delayed
import numpy as np

mne.set_log_level(verbose="CRITICAL")


def get_bad_indices(epochs, tmin=0, tmax=5, threshold=150):
    data = epochs.get_data(units="uV", tmin=tmin, tmax=tmax, picks="eeg")

    I = np.where((np.abs(data) > threshold).any(axis=(1, 2)))[0]

    return I


def export_evoked(subject, dataset_name, save_base):
    mne.set_log_level(verbose="CRITICAL")
    files = os.listdir(Path("./Dataset") / "epochs" / dataset_name)

    files_to_load = []

    for file in files:
        if file.startswith(f"sub-{subject}_"):
            files_to_load.append(file)
    files_to_load = natsort.natsorted(files_to_load)

    print(f"Exporting data for subject {subject}")

    epochs_list = []
    for file in files_to_load:
        e = mne.read_epochs(Path("./Dataset") / "epochs" / dataset_name / file)
        epochs_list.append(e)

    epochs = mne.concatenate_epochs(epochs_list)

    epochs.apply_baseline(baseline=(-0.8, -0.2))

    if dataset_name == "Lee2019_MI":
        tmin, tmax = 0.5, 3.5
    else:
        tmin, tmax = 0.5, 4.5

    indices = get_bad_indices(epochs, tmin=tmin, tmax=tmax, threshold=150)

    # epochs.drop_bad(reject={"eeg": 250e-6})
    epochs.drop(indices=indices)

    if (len(epochs["left_hand"]) == 0) or (len(epochs["right_hand"]) == 0):
        raise RuntimeError(f"No epochs remained for subject {subject}.")

    left_hand = epochs["left_hand"]
    right_hand = epochs["right_hand"]

    left_hand.average().save(save_base / f"sub-{subject}_left_hand.fif", overwrite=True)

    right_hand.average().save(
        save_base / f"sub-{subject}_right_hand.fif", overwrite=True
    )


if __name__ == "__main__":

    DATASETS = ["Dreyer2023", "Lee2019_MI"]
    DATASETS = ["Lee2019_MI"]

    for dataset_name in DATASETS:
        save_base = Path("Dataset") / "evoked" / dataset_name
        save_base.mkdir(exist_ok=True, parents=True)

        module = importlib.import_module("moabb.datasets")
        dataset = getattr(module, dataset_name)()

        # export_meta_data(dataset, save_base)

        if dataset_name == "Dreyer2023":
            dataset.subject_list.remove(40)
            dataset.subject_list.remove(59)

        Parallel(n_jobs=-5)(
            delayed(export_evoked)(
                subject=subject,
                dataset_name=dataset_name,
                save_base=save_base,
            )
            for subject in dataset.subject_list
        )
        exit()

        """
        for subject in dataset.subject_list:
            export_evoked(
                subject=subject,
                dataset=dataset,
                dataset_name=dataset_name,
                save_base=save_base,
            )
        """

    print("\nDone preprocessing all windows.")
