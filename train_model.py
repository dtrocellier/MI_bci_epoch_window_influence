import os
import math
import time
import functools
import pandas as pd
import numpy as np
import torch
from torch import nn
from multiprocessing import Pool
from numpy.random import seed
import random
import wandb

from src.read_data import load_data, loaders_cross
from braindecode.models import Deep4Net
from config import (
    MODELS, DATASETS, DATA_PATH,
    SAVE_PERF_PATH, SWEEP_ID_FILE, SWEEP_PROJECT,
)

import mne
mne.set_log_level(verbose="Warning")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

seed(2002012)
torch.manual_seed(2002012)
random.seed(2002012)
torch.cuda.manual_seed(2002012)
torch.cuda.manual_seed_all(2002012)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

DATASET_PARAMS = {
    'Large': {'n_classes': 2, 'n_chans': 27, 'sfreq': 512, 'input_window_samples': 2048},
}


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    seed(worker_seed)
    random.seed(worker_seed)


def lastConvLengthDeep4Net(n):
    return math.floor((n - 9) / 3)


def build_optimizer(model, optimizer, learning_rate, weight_decay):
    if optimizer == "adamW":
        return torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    elif optimizer == "adam":
        return torch.optim.Adam(model.parameters(), lr=learning_rate)
    raise ValueError(f"Unknown optimizer: {optimizer}")


def build_scheduler(optimizer, scheduler, lr_gamma, lr_step_size, n_epochs=100):
    if scheduler == "CosineAnnealingLR":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs - 1)
    elif scheduler == "StepLR":
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=lr_step_size, gamma=lr_gamma)
    elif scheduler == "ReduceLROnPlateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10)
    raise ValueError(f"Unknown scheduler: {scheduler}")


def build_model(model_name, n_chans, n_classes, input_window_samples):
    if model_name == "Deep4Net":
        final_length = lastConvLengthDeep4Net(lastConvLengthDeep4Net(
            lastConvLengthDeep4Net(lastConvLengthDeep4Net(input_window_samples))))
        model = Deep4Net(
            in_chans=n_chans,
            n_classes=n_classes,
            input_window_samples=input_window_samples,
            final_conv_length=final_length,
        )
    else:
        raise ValueError(f"Unknown model: {model_name}")
    return model.to(device).float()


def train_one_epoch(model, train_loader, optimizer, criterion):
    model.train()
    total_loss, correct = 0, 0
    for data, target in train_loader:
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        correct += output.argmax(dim=1).eq(target).sum().item()
    return total_loss / len(train_loader.dataset), correct / len(train_loader.dataset)


def validate(model, loader, criterion):
    model.eval()
    total_loss, correct = 0, 0
    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            total_loss += criterion(output, target).item()
            correct += output.argmax(dim=1).eq(target).sum().item()
    return total_loss / len(loader.dataset), correct / len(loader.dataset)


def build_dataset(test_subject, batch_size, dataset, epoch_window):
    dict_config = load_dict_config()

    if dataset == 'Large':
        X, Y = load_data(path=DATA_PATH, window=epoch_window)
        train_data, valid_data, test_data = loaders_cross(
            test_subject, X, Y, dataset='Large', reg_subject=False
        )
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    return train_data, valid_data, test_data


def train(model_name, dataset):
    with wandb.init():
        config = wandb.config
        epoch_window = config.epoch_window
        test_subject  = config.test_subject

        wandb.config.update({'model': model_name, 'dataset': dataset}, allow_val_change=True)
        wandb.run.name = f"{model_name}_{dataset}_{epoch_window}_{test_subject}"
        wandb.run.save()

        params = DATASET_PARAMS[dataset]

        train_data, valid_data, test_data = build_dataset(
            test_subject, config.batch_size, dataset, epoch_window
        )
        model = build_model(model_name, params['n_chans'], params['n_classes'], params['input_window_samples'])
        optimizer = build_optimizer(model, config.optimizer, config.lr, config.weight_decay)
        scheduler = build_scheduler(optimizer, config.scheduler, config.lr_gamma, config.lr_step_size, config.n_epochs)
        criterion = nn.NLLLoss()

        best_acc = 0
        save_name = f"{config.n_epochs}_epochs_{config.batch_size}_batch_size_{config.lr}_lr"
        best_model_path = f"model/{model_name}/{model_name}_{dataset}_{epoch_window}_{test_subject}_{save_name}.pt"
        os.makedirs(os.path.dirname(best_model_path), exist_ok=True)

        for epoch in range(config.n_epochs):
            t_start = time.time()
            train_loss, train_acc = train_one_epoch(model, train_data, optimizer, criterion)
            valid_loss, valid_acc = validate(model, valid_data, criterion)
            duration = time.time() - t_start

            if config.scheduler == "ReduceLROnPlateau":
                scheduler.step(valid_loss)
            else:
                scheduler.step()

            if valid_acc > best_acc:
                best_acc = valid_acc
                torch.save(model.state_dict(), best_model_path)

            wandb.log({
                "train_loss": train_loss, "train_accuracy": train_acc,
                "valid_loss": valid_loss, "valid_accuracy": valid_acc,
                "best_valid_accuracy": best_acc,
                "time_one_epoch": duration,
                "lr": optimizer.param_groups[0]['lr'],
            })

        model.load_state_dict(torch.load(best_model_path))
        test_loss, test_acc = validate(model, test_data, criterion)
        wandb.log({"test_loss": test_loss, "test_accuracy": test_acc})

        _save_results(test_subject, dataset, model_name, epoch_window, test_acc)
        print(f"Done: {model_name} | {dataset} | {epoch_window} | subject {test_subject} → {test_acc:.4f}")


def _save_results(subject, dataset, model_name, epoch_window, accuracy):
    csv_path = SAVE_PERF_PATH
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
    else:
        df = pd.DataFrame(columns=["subject", "dataset", "model", "epoch_window", "accuracy"])

    mask = (
        (df["subject"] == subject) &
        (df["dataset"] == dataset) &
        (df["model"] == model_name) &
        (df["epoch_window"] == epoch_window)
    )
    if mask.sum() > 0:
        df.loc[mask, "accuracy"] = accuracy
    else:
        df = pd.concat([df, pd.DataFrame([{
            "subject": subject, "dataset": dataset,
            "model": model_name, "epoch_window": epoch_window,
            "accuracy": accuracy,
        }])], ignore_index=True)

    df.to_csv(csv_path, index=False)


def load_dict_config():
    return {
        'path': DATA_PATH,
        'reg_subject': False,
        'save_perf_path': SAVE_PERF_PATH,
        'sweep_directory': SWEEP_ID_FILE,
        'sweep_project_name': SWEEP_PROJECT,
    }


def main(gpu_id=0):
    global device

    if torch.cuda.device_count() > 1:
        import subprocess
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=memory.free', '--format=csv,nounits,noheader', '-i', str(gpu_id)],
            stdout=subprocess.PIPE,
        )
        available_memory = int(result.stdout.decode('utf-8').strip())
        if available_memory > 8000:
            device = f"cuda:{gpu_id}"
            torch.cuda.set_device(gpu_id)
            print(f"GPU {gpu_id}: {available_memory}MB available")
        else:
            print(f"GPU {gpu_id} has insufficient memory ({available_memory}MB), skipping.")
            return
    else:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    with open(SWEEP_ID_FILE, 'r') as f:
        sweep_id = f.read().strip()
    print(f"Sweep ID: {sweep_id}")

    for dataset in DATASETS:
        for model_name in MODELS:
            print(f"\nRunning sweep — model: {model_name}  dataset: {dataset}")
            wandb.agent(
                sweep_id,
                functools.partial(train, model_name=model_name, dataset=dataset),
                project=SWEEP_PROJECT,
            )


if __name__ == "__main__":
    num_gpus = torch.cuda.device_count()
    print(f"Found {num_gpus} GPU(s)")

    if num_gpus > 1:
        with Pool(num_gpus) as p:
            p.map(main, range(num_gpus))
    else:
        main(0)