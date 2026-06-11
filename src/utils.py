import importlib
import json
import math
import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset

from captum.attr import IntegratedGradients, DeepLift, LRP


def load_config():
    base = Path(__file__).parent.parent.resolve()
    with open(base / "conf" / "config.yaml", "r") as f:
        config = yaml.safe_load(f)
    return config


def window_suffix(tmin, tmax, sfreq):
    return f"{tmin}_{tmax}_{sfreq}"


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_all_runs(subject, cfg):
    base = Path(cfg.dataset.path)

    sfreq = cfg.model.sfreq
    tmin = cfg.epoch_window.tmin
    tmax = cfg.epoch_window.tmax

    suffix = f"{tmin}_{tmax}_{sfreq}"

    runs = cfg.dataset.runs.train + cfg.dataset.runs.test
    sessions = list(range(1, cfg.dataset.n_sessions + 1))

    X_list, y_list = [], []
    for session in sessions:
        for run in runs:
            X = np.load(base / f"sub-{subject}_ses-{session}_run-{run}_X_{suffix}.npy")
            y = np.load(base / f"sub-{subject}_ses-{session}_run-{run}_y_{suffix}.npy")

            X_list.append(X)
            y_list.append(y)

    X = np.concatenate(X_list)
    y = np.concatenate(y_list)

    return X, y


def load_test_runs(subject, cfg):
    base = Path(cfg.dataset.path)

    sfreq = cfg.model.sfreq
    tmin = cfg.epoch_window.tmin
    tmax = cfg.epoch_window.tmax

    suffix = f"{tmin}_{tmax}_{sfreq}"

    runs = cfg.dataset.runs.test
    sessions = list(range(1, cfg.dataset.n_sessions + 1))

    X_list, y_list = [], []
    for session in sessions:
        for run in runs:
            X = np.load(base / f"sub-{subject}_ses-{session}_run-{run}_X_{suffix}.npy")
            y = np.load(base / f"sub-{subject}_ses-{session}_run-{run}_y_{suffix}.npy")

            X_list.append(X)
            y_list.append(y)

    X = np.concatenate(X_list)
    y = np.concatenate(y_list)

    return X, y


def build_dataset(cfg):
    # make a list of all subject
    subjects = list(range(1, cfg.dataset.subjects.n_subjects + 1))
    if cfg.dataset.subjects.exclude is not None:
        for subject in cfg.dataset.subjects.exclude:
            subjects.remove(subject)

    # remove test subject
    test_subject = cfg.subject
    subjects.remove(test_subject)

    # split the dataset into train and valid
    test_size = cfg.split.test_size
    random_state = cfg.split.random_state
    train_subjects, valid_subjects = train_test_split(
        subjects, test_size=test_size, random_state=random_state
    )

    if cfg.debug:
        train_subjects = train_subjects[:10]
        valid_subjects = valid_subjects[:10]

    # load training data
    X_list, y_list = [], []
    for subject in train_subjects:
        X, y = load_all_runs(subject, cfg)
        X_list.append(X)
        y_list.append(y)
    X_train = np.concatenate(X_list)
    y_train = np.concatenate(y_list)

    X_train = torch.tensor(X_train, dtype=torch.float)
    y_train = torch.tensor(y_train, dtype=torch.long)

    # load validation data
    X_list, y_list = [], []
    for subject in valid_subjects:
        X, y = load_test_runs(subject, cfg)
        X_list.append(X)
        y_list.append(y)
    X_valid = np.concatenate(X_list)
    y_valid = np.concatenate(y_list)

    X_valid = torch.tensor(X_valid, dtype=torch.float)
    y_valid = torch.tensor(y_valid, dtype=torch.long)

    # load test data
    X_test, y_test = load_test_runs(test_subject, cfg)

    X_test = torch.tensor(X_test, dtype=torch.float)
    y_test = torch.tensor(y_test, dtype=torch.long)

    # Normalization

    _, n_channels, n_times = X_train.shape
    mean = X_train.transpose(1, 2).reshape(-1, n_channels).mean(dim=0)
    std = X_train.transpose(1, 2).reshape(-1, n_channels).std(dim=0)

    X_train = (X_train - mean.unsqueeze(0).unsqueeze(2)) / std.unsqueeze(0).unsqueeze(2)
    X_valid = (X_valid - mean.unsqueeze(0).unsqueeze(2)) / std.unsqueeze(0).unsqueeze(2)
    X_test = (X_test - mean.unsqueeze(0).unsqueeze(2)) / std.unsqueeze(0).unsqueeze(2)

    train_loader = torch.utils.data.DataLoader(
        list(zip(torch.unbind(X_train), torch.unbind(y_train))),
        batch_size=cfg.model.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=0,
    )
    valid_loader = torch.utils.data.DataLoader(
        list(zip(torch.unbind(X_valid), torch.unbind(y_valid))),
        batch_size=cfg.model.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=0,
    )
    test_loader = torch.utils.data.DataLoader(
        list(zip(torch.unbind(X_test), torch.unbind(y_test))),
        batch_size=cfg.model.batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=0,
    )

    return train_loader, valid_loader, test_loader


class PosDataset(Dataset):
    def __init__(self, X, y, pos):
        self.X = X
        self.y = y
        self.pos = pos

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx], self.pos


def build_dataset_REVE(cfg):
    # make a list of all subject
    subjects = list(range(1, cfg.dataset.subjects.n_subjects + 1))
    if cfg.dataset.subjects.exclude is not None:
        for subject in cfg.dataset.subjects.exclude:
            subjects.remove(subject)

    # remove test subject
    test_subject = cfg.subject
    subjects.remove(test_subject)

    # split the dataset into train and valid
    test_size = cfg.split.test_size
    random_state = cfg.split.random_state
    train_subjects, valid_subjects = train_test_split(
        subjects, test_size=test_size, random_state=random_state
    )

    if cfg.debug:
        train_subjects = train_subjects[:10]
        valid_subjects = valid_subjects[:10]

    # load training data
    X_list, y_list = [], []
    for subject in train_subjects:
        X, y = load_all_runs(subject, cfg)
        X_list.append(X)
        y_list.append(y)
    X_train = np.concatenate(X_list)
    y_train = np.concatenate(y_list)

    X_train = torch.tensor(X_train, dtype=torch.float)
    y_train = torch.tensor(y_train, dtype=torch.long)

    # load validation data
    X_list, y_list = [], []
    for subject in valid_subjects:
        X, y = load_test_runs(subject, cfg)
        X_list.append(X)
        y_list.append(y)
    X_valid = np.concatenate(X_list)
    y_valid = np.concatenate(y_list)

    X_valid = torch.tensor(X_valid, dtype=torch.float)
    y_valid = torch.tensor(y_valid, dtype=torch.long)

    # load test data
    X_test, y_test = load_test_runs(test_subject, cfg)

    X_test = torch.tensor(X_test, dtype=torch.float)
    y_test = torch.tensor(y_test, dtype=torch.long)

    # get channel pos
    from braindecode.models import REVE

    model = REVE.from_pretrained(
        "brain-bzh/reve-base",
        n_outputs=1,
        n_chans=27,
        n_times=200,
    )  # just for getting the channel pos

    with open(Path(cfg.dataset.path) / "meta_data.json", "r") as f:
        ch_names = json.load(f)["ch_names"]

    pos = model.get_positions(ch_names)

    # Normalization

    _, n_channels, n_times = X_train.shape
    mu = X_train.mean()
    std = X_train.std()

    X_train = (X_train - mu) / std
    X_valid = (X_valid - mu) / std
    X_test = (X_test - mu) / std

    # clipping (as the original REVE pretraining pipeline)
    std = X_train.std()
    X_train = torch.clamp(X_train, min=-15 * std, max=15 * std)
    X_valid = torch.clamp(X_valid, min=-15 * std, max=15 * std)
    X_test = torch.clamp(X_test, min=-15 * std, max=15 * std)

    # build datasets

    train_set = PosDataset(X_train, y_train, pos)
    valid_set = PosDataset(X_valid, y_valid, pos)
    test_set = PosDataset(X_test, y_test, pos)

    # build dataloader

    train_loader = DataLoader(train_set, batch_size=cfg.model.batch_size, shuffle=True)
    valid_loader = DataLoader(valid_set, batch_size=cfg.model.batch_size, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=cfg.model.batch_size, shuffle=False)

    return train_loader, valid_loader, test_loader


def lastConvLengthDeep4Net(n):
    return math.floor((n - 9) / 3)


def build_model(cfg):
    sfreq = cfg.model.sfreq

    if sfreq is None:
        sfreq = "none"

    module = importlib.import_module("braindecode.models")
    n_times = cfg.dataset.input_window_samples[sfreq]
    n_chans = cfg.dataset.n_channels
    n_outputs = cfg.dataset.n_classes

    match cfg.model.name:
        case "Deep4Net":
            final_length = lastConvLengthDeep4Net(
                lastConvLengthDeep4Net(
                    lastConvLengthDeep4Net(lastConvLengthDeep4Net(n_times))
                )
            )
            model = getattr(module, cfg.model.name)(
                n_chans=n_chans,
                n_outputs=n_outputs,
                n_times=n_times,
                # final_conv_length=final_length,
            )
        case "REVE":
            from braindecode.models import REVE

            model = REVE.from_pretrained(
                "brain-bzh/reve-base",
                n_chans=n_chans,
                n_outputs=n_outputs,
                n_times=n_times,
            )
        case _:
            model = getattr(module, cfg.model.name)(
                n_chans=n_chans,
                n_outputs=n_outputs,
                n_times=n_times,
            )

    return model


def build_optimizer(model, cfg_optimizer):
    module = importlib.import_module(cfg_optimizer.module)

    if cfg_optimizer.kwargs is None:
        kwargs = {}
    else:
        kwargs = cfg_optimizer.kwargs

    optimizer = getattr(module, cfg_optimizer.name)(
        filter(lambda p: p.requires_grad, model.parameters()),
        **kwargs,
    )
    return optimizer


def build_scheduler(optimizer, cfg_scheduler):
    module = importlib.import_module(cfg_scheduler.module)
    scheduler = getattr(module, cfg_scheduler.name)(optimizer, **cfg_scheduler.kwargs)
    return scheduler


def build_criterion(cfg_criterion):
    kwargs = cfg_criterion.kwargs
    if kwargs is None:
        kwargs = {}
    module = importlib.import_module(cfg_criterion.module)
    criterion = getattr(module, cfg_criterion.name)(**kwargs)
    return criterion


def get_func_build_dataset(path):
    module_name, func_name = path.rsplit(".", 1)
    module = importlib.import_module(module_name)
    func = getattr(module, func_name)

    return func


def train_one_epoch(model, train_loader, optimizer, criterion, device):
    model.train()
    total_loss, correct = 0, 0
    for batch in train_loader:
        if len(batch) == 3:
            data, target, pos = batch
            pos = pos.to(device)
        else:
            data, target = batch
            pos = None

        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()

        if pos is not None:
            output = model(data, pos=pos)
        else:
            output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        correct += output.argmax(dim=1).eq(target).sum().item()
    return total_loss / len(train_loader.dataset), correct / len(train_loader.dataset)


def validate(model, loader, criterion, device):
    model.eval()
    total_loss, correct = 0, 0
    with torch.no_grad():
        for batch in loader:
            if len(batch) == 3:
                data, target, pos = batch
                pos = pos.to(device)
            else:
                data, target = batch
                pos = None

            data, target = data.to(device), target.to(device)

            if pos is not None:
                output = model(data, pos=pos)
            else:
                output = model(data)

            total_loss += criterion(output, target).item()
            correct += output.argmax(dim=1).eq(target).sum().item()
    return total_loss / len(loader.dataset), correct / len(loader.dataset)


def save_results(cfg, accuracy):
    base = Path(cfg.path.results)
    csv_path = base / "classification_results.csv"
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
    else:
        base.mkdir(exist_ok=True, parents=True)
        df = pd.DataFrame(
            columns=["subject", "dataset", "model", "epoch_window", "accuracy"]
        )

    subject = cfg.subject
    dataset = cfg.dataset.name
    model_name = cfg.model.name
    epoch_window = cfg.epoch_window.name

    mask = (
            (df["subject"] == subject)
            & (df["dataset"] == dataset)
            & (df["model"] == model_name)
            & (df["epoch_window"] == epoch_window)
    )
    if mask.sum() > 0:
        df.loc[mask, "accuracy"] = accuracy
    else:
        df = pd.concat(
            [
                df,
                pd.DataFrame(
                    [
                        {
                            "subject": subject,
                            "dataset": dataset,
                            "model": model_name,
                            "epoch_window": epoch_window,
                            "accuracy": accuracy,
                        }
                    ]
                ),
            ],
            ignore_index=True,
        )

    df.to_csv(csv_path, index=False)


def get_data_from_loader(loader):
    X_list = []
    y_list = []
    pos_list = []

    for idx, batch in enumerate(loader):

        if len(batch) == 3:
            X, y, pos = batch
        else:
            X, y = batch
            pos = None

        X_list.append(X)
        pos_list.append(pos)
        y_list.append(y)

    X = torch.concat(X_list, dim=0)
    y = torch.concat(y_list, dim=0)

    if pos_list[0] is not None:
        pos = torch.concat(pos_list, dim=0)
    else:
        pos = None

    return X, y, pos


def saliency_map(model, loader, device, class_index=1):
    model.eval()

    for batch in loader:
        if len(batch) == 3:
            X, _, _ = batch
        else:
            X, _ = batch
        break

    saliency = torch.zeros((X.shape[1], X.shape[2]), device=device)

    n_used = 0
    for batch in loader:

        if len(batch) == 3:
            data, target, pos = batch
            pos = pos.to(device)
        else:
            data, target = batch
            pos = None

        mask = target == class_index
        data = data[mask]

        if mask.sum() == 0:
            continue

        n_used += data.shape[0]

        data = data.to(device)
        data.requires_grad = True

        if pos is not None:
            output = model(data, pos=pos[mask])
        else:
            output = model(data)

        output = torch.sum(output, dim=0) / output.shape[0]

        # right
        output_right = output[class_index]
        # If output is not a scalar, consider using torch.sum(output).backward()
        output_right.backward()
        # Assuming data.grad is not None and has the same shape as data
        if data.grad is not None:
            saliency += data.grad.abs().sum(dim=0)  # Sum over the batch
        else:
            raise ValueError("data.grad is None")

    saliency = saliency / n_used
    saliency = saliency.cpu().numpy()

    return {
        "saliency": saliency,
    }


def integrated_gradients_map(model, loader, device, class_index=1):
    return _captum_map(
        model=model,
        loader=loader,
        device=device,
        class_index=class_index,
        method="integrated_gradients",
    )


def deeplift_map(model, loader, device, class_index=1):
    return _captum_map(
        model=model,
        loader=loader,
        device=device,
        class_index=class_index,
        method="deeplift",
    )


def lrp_map(model, loader, device, class_index=1):
    return _captum_map(
        model=model,
        loader=loader,
        device=device,
        class_index=class_index,
        method="lrp",
    )


def _captum_map(model, loader, device, class_index=1, method="integrated_gradients"):
    model.eval()
    from captum.attr._utils.lrp_rules import EpsilonRule
    from braindecode.modules.layers import Ensure4d

    for batch in loader:
        if len(batch) == 3:
            X, _, _ = batch
        else:
            X, _ = batch
        break

    attr_map = torch.zeros((X.shape[1], X.shape[2]), device=device)
    n_samples = 0

    if method == "integrated_gradients":
        attr_method = IntegratedGradients(model)
    elif method == "deeplift":
        attr_method = DeepLift(model)
    elif method == "lrp":
        attr_method = LRP(model)
    else:
        raise ValueError(f"Unknown method: {method}")

    for batch in loader:
        if len(batch) == 3:
            data, target, pos = batch
            pos = pos.to(device)
        else:
            data, target = batch
            pos = None

        mask = target == class_index

        if mask.sum() == 0:
            continue

        data = data[mask].to(device)
        data.requires_grad_(True)
        n_samples += data.shape[0]

        if pos is not None:
            pos = pos[mask].to(device)
            forward_args = (pos,)
        else:
            forward_args = None

        if method in ["integrated_gradients"]:
            baseline = torch.zeros_like(data)

            attr = attr_method.attribute(
                data,
                baselines=baseline,
                target=class_index,
                additional_forward_args=forward_args,
                n_steps=16,
                internal_batch_size=1,
            )
        elif method in ["deeplift"]:
            baseline = torch.zeros_like(data)
            attr = attr_method.attribute(
                data,
                baselines=baseline,
                target=class_index,
                additional_forward_args=forward_args,
            )
        else:
            attr = attr_method.attribute(
                data,
                target=class_index,
                additional_forward_args=forward_args,
            )

        attr_map += attr.abs().sum(dim=0)

    attr_map = attr_map / n_samples

    return {
        method: attr_map.detach().cpu().numpy(),
    }


class BestScore:

    def __init__(self, mode="max", min_delta=0.0):
        """
        Parameters
        ----------
        mode : {"max", "min"}
            Optimization direction.

            - "max": larger values indicate better performance.
            - "min": smaller values indicate better performance.

        min_delta : float, default=0.0
            Minimum change required to qualify as an improvement.
        """
        if mode not in ("max", "min"):
            raise ValueError("mode must be 'max' or 'min'")

        self.mode = mode
        self.min_delta = min_delta
        self.initialize()

    def initialize(self):
        if self.mode == "max":
            self.best_score = float("-inf")
        else:
            self.best_score = float("inf")

    def __call__(self, score):
        return self.step(score)

    def step(self, score):
        if self.mode == "max":
            improved = score > (self.best_score + self.min_delta)
        else:
            improved = score < (self.best_score - self.min_delta)

        if improved:
            self.best_score = score
            return True

        return False


class EarlyStopping:

    def __init__(self, mode="min", patience=3, min_delta=0.0, warmup=0):
        self.mode = mode
        self.patience = patience
        self.min_delta = min_delta
        self.warmup = warmup

        self.best_score = BestScore(mode=mode, min_delta=min_delta)
        self.counter = 0
        self.epoch = 0

    def initialize(self):
        self.best_score.initialize()
        self.counter = 0
        self.epoch = 0

    def __call__(self, score):
        return self.step(score)

    def step(self, score):
        self.epoch += 1

        improved = self.best_score(score)

        if self.epoch <= self.warmup:
            return False

        if improved:
            self.counter = 0
        else:
            self.counter += 1

        return self.counter >= self.patience
