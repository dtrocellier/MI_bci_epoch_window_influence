# Adding a New Dataset or Model

## Adding a New Dataset

### 1. Preprocess the raw data

Write a preprocessing script (or extend `preprocess_data.py`) that produces the following files under `Dataset/<DatasetName>/` for each epoch window:

```
Dataset/<DatasetName>/
├── X_s_<window>.pt     # list[list[Tensor]]  — shape (n_subjects, n_sessions, n_trials, n_chans, n_samples)
└── Y_s_<window>.pt     # list[list[Tensor]]  — shape (n_subjects, n_sessions, n_trials)
```

The window suffix comes from `window_suffix(tmin, tmax)` in `config.py` (e.g. `0_4`, `0.5_4.5`).

---

### 2. Register the dataset name — `config.py`

```python
DATASETS = ['Large', 'MyDataset']   # add your name here
```

---

### 3. Add signal parameters — `train_model.py:33`

```python
DATASET_PARAMS = {
    'Large':     {'n_classes': 2, 'n_chans': 27,  'sfreq': 512, 'input_window_samples': 2048},
    'MyDataset': {'n_classes': 2, 'n_chans': 64,  'sfreq': 250, 'input_window_samples': 1000},
}
```

---

### 4. Add a loading branch — `train_model.py:108` (`build_dataset`)

```python
def build_dataset(test_subject, batch_size, dataset, epoch_window):
    if dataset == 'Large':
        X, Y = load_data(path=DATA_PATH, window=epoch_window)
        train_data, valid_data, test_data = loaders_cross(
            test_subject, X, Y, dataset='Large', reg_subject=False
        )
    elif dataset == 'MyDataset':
        X, Y = load_data(path=DATA_PATH, window=epoch_window, dataset='MyDataset')
        train_data, valid_data, test_data = loaders_cross(
            test_subject, X, Y, dataset='MyDataset', reg_subject=False
        )
    else:
        raise ValueError(f"Unknown dataset: {dataset}")
    return train_data, valid_data, test_data
```

---

### 5. Update `load_data` — `src/read_data.py:287`

The current implementation hardcodes `dataset = 'Large'`. Add a `dataset` parameter:

```python
def load_data(path='Dataset', window='0_4', dataset='Large'):
    X = torch.load(path + '/' + dataset + '/X_s_' + window + '.pt')
    Y = torch.load(path + '/' + dataset + '/Y_s_' + window + '.pt')
    return X, Y
```

---

### 6. Check `loaders_cross` compatibility — `src/read_data.py:443`

`loaders_cross` uses the `dataset` name only to pick `sep` (the train/test session split index) and `batch_size`:

```python
sep        = 1 if dataset in ['BNCI', 'BNCI2'] else 2
batch_size = 288 if dataset == 'BNCI' else 256
```

If your dataset has a different number of calibration sessions, add it to the appropriate condition.

---

---

## Adding a New Model

### 1. Register the model name — `config.py`

```python
MODELS = ['Deep4Net', 'MyModel']   # add your name here
```

---

### 2. Add a build branch — `train_model.py:66` (`build_model`)

**If the model is from braindecode**, import it at the top of `train_model.py` and add a branch:

```python
from braindecode.models import Deep4Net, EEGNetv4   # add import

def build_model(model_name, n_chans, n_classes, input_window_samples):
    if model_name == "Deep4Net":
        ...
    elif model_name == "EEGNetv4":
        model = EEGNetv4(
            in_chans=n_chans,
            n_classes=n_classes,
            input_window_samples=input_window_samples,
            final_conv_length='auto',
        )
    else:
        raise ValueError(f"Unknown model: {model_name}")
    return model.to(device).float()
```

**If the model is custom**, add its class to `src/models.py` (create the file if needed), import it, and add its branch the same way:

```python
from src.models import MyModel

def build_model(...):
    ...
    elif model_name == "MyModel":
        model = MyModel(n_chans, n_classes, input_window_samples)
    ...
```

---

### 3. Verify the output format

`train_one_epoch` and `validate` expect the model to output **log-softmax probabilities** (compatible with `NLLLoss`). Make sure your model's `forward()` ends with `F.log_softmax(x, dim=1)`.

---

## Checklist

### New dataset

- [ ] Preprocessed tensors saved as `Dataset/<Name>/X_s_<window>.pt` and `Y_s_<window>.pt`
- [ ] Name added to `DATASETS` in `config.py`
- [ ] Entry added to `DATASET_PARAMS` in `train_model.py`
- [ ] Branch added in `build_dataset()` in `train_model.py`
- [ ] `load_data()` in `src/read_data.py` accepts the new dataset name
- [ ] `loaders_cross()` session split (`sep`) is correct for the new dataset

### New model

- [ ] Name added to `MODELS` in `config.py`
- [ ] Branch added in `build_model()` in `train_model.py`
- [ ] Model output is log-softmax (compatible with `NLLLoss`)