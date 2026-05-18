


import mne
from mne.preprocessing import EOGRegression
import numpy as np
from tqdm import tqdm


def preprocess(raw, steps={}):
    assert isinstance(steps, dict)
    raw.load_data()
    if "remove_eog_artifact" in steps:
        mne.set_eeg_reference(raw, ref_channels=[
            'Fz', 'FCz', 'Cz', 'CPz', 'Pz', 'C1', 'C3', 'C5', 'C2', 'C4', 'C6',
            'F4', 'FC2', 'FC4', 'FC6', 'CP2', 'CP4', 'CP6', 'P4',
            'F3', 'FC1', 'FC3', 'FC5', 'CP1', 'CP3', 'CP5', 'P3'
        ], copy=False, ch_type="eeg")
        raw.set_channel_types({"EOG1": "eog", "EOG2": "eog", "EOG3": "eog"})
        weights = EOGRegression().fit(raw)
        weights.apply(raw, copy=False)
    if "drop_channels" in steps:
        for channel in steps["drop_channels"]:
            if channel in raw.ch_names:
                raw.drop_channels(channel)
    if "filter" in steps:
        assert isinstance(steps["filter"], list)
        raw.filter(steps["filter"][0], steps["filter"][1])
    return raw


class Epoching:

    def __init__(self, dic_raw_data, steps_preprocess=None, key_session=[], key_events={"769": 0, "770": 1}):
        self.tmin = steps_preprocess["tmin"]
        self.tmax = steps_preprocess["tmax"]
        self.length_epoch = steps_preprocess["lenght"]
        self.overlap = steps_preprocess["overlap"]
        self.n_events_per_trial = steps_preprocess["n_events_per_trial"]
        self.steps_preprocess = steps_preprocess
        self.key_session = key_session
        self.key_events = key_events
        self.dic_raw_data = dic_raw_data

        if "n_chans" in steps_preprocess:
            self.n_chans = steps_preprocess["n_chans"]
        else:
            self.n_chans = len(self.dic_raw_data[key_session[0]].ch_names)
            if "drop_channels" in steps_preprocess:
                for channel in self.dic_raw_data[key_session[0]].ch_names:
                    if channel in steps_preprocess["drop_channels"]:
                        self.n_chans -= 1

        self.list_start = np.arange(self.tmin, (self.tmax + self.overlap) - self.length_epoch, self.overlap)
        self.list_stop = np.arange(self.tmin + self.length_epoch, (self.tmax + self.overlap), self.overlap)
        self.time_step = int(self.length_epoch * self.dic_raw_data[key_session[0]].info['sfreq'])
        self.n_events = len(self.list_start) * self.n_events_per_trial * len(key_session)

        self.X = np.zeros((self.n_events, self.n_chans, self.time_step))
        self.Y = np.zeros((self.n_events))

    def run(self):
        i = 0
        for key in tqdm(self.key_session, desc="epoching"):
            if self.steps_preprocess is not None:
                preprocess(self.dic_raw_data[key], self.steps_preprocess)
            epoch = mne.Epochs(
                self.dic_raw_data[key],
                mne.events_from_annotations(self.dic_raw_data[key], self.key_events)[0],
                tmin=-1, tmax=5, baseline=(None, 0),
            )
            assert len(epoch.events[:, 2]) == self.n_events_per_trial, (
                f"{key} doesn't have {self.n_events_per_trial} events, got {len(epoch.events[:, 2])}"
            )
            for start, stop in zip(self.list_start, self.list_stop):
                self.X[i: i + self.n_events_per_trial] = epoch.get_data(tmin=start, tmax=stop)
                self.Y[i: i + self.n_events_per_trial] = epoch.events[:, 2]
                i += self.n_events_per_trial
        return self.X, self.Y
