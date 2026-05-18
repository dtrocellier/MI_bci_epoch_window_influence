import mne
import os
import os.path as osp
import numpy as np
from tqdm import tqdm
import torch

import pickle

from config import EPOCH_WINDOWS, DATA_PATH, window_suffix
from src.utils import preprocess

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
mne.set_log_level(verbose="Warning")


def read_data(init_path):
    files_dir = os.listdir(init_path)
    for file in files_dir:
        if "." in file:
            files_dir.remove(file)
    files_dir.sort()

    participant_dir = [os.listdir(osp.join(init_path, files_dir[i])) for i in range(len(files_dir))]
    for list_participant in participant_dir:
        list_participant.sort()

    print("Successfully accessed directory:", init_path)
    print(files_dir)
    print(participant_dir)
    return participant_dir, files_dir


def collect_data(files_dir, participant_dir, init_path):
    dic_data = {}
    for i in range(len(files_dir)):
        for j in range(len(participant_dir[i])):
            dic_data[participant_dir[i][j] + "_1"] = mne.io.read_raw_gdf(
                osp.join(init_path, files_dir[i], participant_dir[i][j], participant_dir[i][j] + "_R1_acquisition.gdf"), verbose="CRITICAL")
            dic_data[participant_dir[i][j] + "_2"] = mne.io.read_raw_gdf(
                osp.join(init_path, files_dir[i], participant_dir[i][j], participant_dir[i][j] + "_R2_acquisition.gdf"), verbose="CRITICAL")
            dic_data[participant_dir[i][j] + "_3"] = mne.io.read_raw_gdf(
                osp.join(init_path, files_dir[i], participant_dir[i][j], participant_dir[i][j] + "_R3_onlineT.gdf"), verbose="CRITICAL")
            dic_data[participant_dir[i][j] + "_4"] = mne.io.read_raw_gdf(
                osp.join(init_path, files_dir[i], participant_dir[i][j], participant_dir[i][j] + "_R4_onlineT.gdf"), verbose="CRITICAL")
            try:
                dic_data[participant_dir[i][j] + "_5"] = mne.io.read_raw_gdf(
                    osp.join(init_path, files_dir[i], participant_dir[i][j], participant_dir[i][j] + "_R5_onlineT.gdf"), verbose="CRITICAL")
            except FileNotFoundError:
                pass
            try:
                dic_data[participant_dir[i][j] + "_6"] = mne.io.read_raw_gdf(
                    osp.join(init_path, files_dir[i], participant_dir[i][j], participant_dir[i][j] + "_R6_onlineT.gdf"), verbose="CRITICAL")
            except FileNotFoundError:
                pass
    return dic_data


def extract_keys(all_subject):
    keys = []
    for subj in all_subject:
        if subj == "A59":
            keys += [subj + "_" + str(i) for i in range(1, 5)]
        else:
            keys += [subj + "_" + str(i) for i in range(1, 7)]
    return keys



def epoching(dic_data, key_subject=[], steps_epoching=None, key_events={"769": 0, "770": 1}):
    tmin              = steps_epoching["tmin"]
    tmax              = steps_epoching["tmax"]
    n_events_per_trial = steps_epoching["n_events_per_trial"]

    X_list = []
    Y_list = []
    for key_s in tqdm(key_subject, desc="epoching"):
        X = []
        Y = []
        for key in extract_keys([key_s]):
            epoch = mne.Epochs(
                dic_data[key],
                mne.events_from_annotations(dic_data[key], key_events)[0],
                tmin=-1, tmax=5, baseline=(None, 0)
            )
            X.append(epoch.get_data(tmin=tmin, tmax=tmax))
            Y.append(epoch.events[:, 2])
        X_list.append(X)
        Y_list.append(Y)
    return X_list, Y_list


def prepro_Y_s(Y_):
    Y_list = []
    for Y_sub in Y_:
        Y_list_sub = []
        for Y_sess in Y_sub:
            Y_list_sub.append(torch.from_numpy(Y_sess).long())
        Y_list.append(Y_list_sub)
    return Y_list


# ── Main ──────────────────────────────────────────────────────────────────────

init_path = DATA_PATH + "Big_dataset"
out_path  = DATA_PATH + "Big_dataset"
os.makedirs(out_path, exist_ok=True)

participant_dir, files_dir = read_data(init_path)
dic_data = collect_data(files_dir, participant_dir, init_path)
session  = [subj for sess in participant_dir for subj in sess]

# Preprocessing is window-independent: apply once
steps_preprocess = {
    "filter": [0.5, 40],
    "drop_channels": ['EOG1', 'EOG2', 'EOG3', 'EMGg', 'EMGd'],
}
for raw in tqdm(dic_data.values(), desc="preprocess"):
    preprocess(raw, steps_preprocess)

# Save channel names once (same for all windows)
last_raw = next(iter(dic_data.values()))
with open(out_path + '/channels.pkl', "wb") as f:
    pickle.dump(last_raw.info.ch_names, f)

# Save order once
with open(out_path + '/order.pkl', "wb") as f:
    pickle.dump(session, f)

# Loop over epoch windows defined in config.py
for tmin, tmax in EPOCH_WINDOWS:
    suffix = window_suffix(tmin, tmax)
    print(f"\n=== Epoch window: tmin={tmin}s  tmax={tmax}s  (suffix: {suffix}) ===")

    steps_epoching = {
        "tmin": tmin,
        "tmax": tmax,
        "overlap": 1,
        "lenght": tmax - tmin,
        "n_events_per_trial": 40,
        "one_hot": False,
    }

    X, Y = epoching(dic_data, session, steps_epoching)

    with open(out_path + f'/X_npy_{suffix}.pkl', "wb") as f:
        pickle.dump(X, f)
    with open(out_path + f'/Y_npy_{suffix}.pkl', "wb") as f:
        pickle.dump(Y, f)

    list_xs = [[torch.from_numpy(sess).float() for sess in sub] for sub in X]
    list_ys = prepro_Y_s(Y)

    torch.save(list_xs, out_path + f'/X_s_{suffix}.pt')
    torch.save(list_ys, out_path + f'/Y_s_{suffix}.pt')

    list_x = [torch.cat(sublist) for sublist in list_xs]
    list_y = [torch.cat(sublist) for sublist in list_ys]
    torch.save(list_x, out_path + f'/X_{suffix}.pt')
    torch.save(list_y, out_path + f'/Y_{suffix}.pt')

    print(f"Saved X_s_{suffix}.pt, Y_s_{suffix}.pt to {out_path}")

print("\nDone preprocessing all windows.")