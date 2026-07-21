from pathlib import Path
import importlib
import mne
import os
import natsort

mne.set_log_level(verbose="CRITICAL")


def export_evoked(dataset, dataset_name, save_base):
    for subject in dataset.subject_list:
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

        left_hand = epochs["left_hand"]
        right_hand = epochs["right_hand"]

        left_hand.average().save(
            save_base / f"sub-{subject}_left_hand.fif", overwrite=True
        )

        right_hand.average().save(
            save_base / f"sub-{subject}_right_hand.fif", overwrite=True
        )


if __name__ == "__main__":

    DATASETS = ["Dreyer2023", "Lee2019_MI"]
    # DATASETS = ["Lee2019_MI"]

    for dataset_name in DATASETS:
        save_base = Path("Dataset") / "evoked" / dataset_name
        save_base.mkdir(exist_ok=True, parents=True)

        module = importlib.import_module("moabb.datasets")
        dataset = getattr(module, dataset_name)()

        # export_meta_data(dataset, save_base)

        export_evoked(
            dataset=dataset,
            dataset_name=dataset_name,
            save_base=save_base,
        )

    print("\nDone preprocessing all windows.")
