import gc
import os
import time
from pathlib import Path

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import hydra
import torch
import wandb
from omegaconf import OmegaConf

from src.utils import (
    get_func_build_dataset,
    build_model,
    build_optimizer,
    build_scheduler,
    build_criterion,
    train_one_epoch,
    validate,
    set_seed,
    save_results,
    saliency_map,
    integrated_gradients_map,
    deeplift_map,
    EarlyStopping,
    clear_cuda,
)


def run(cfg):
    print(cfg)

    run_name = f"{cfg.dataset.name}_{cfg.model.sname}_{cfg.epoch_window.name}_sub-{cfg.subject}"

    device = "cuda" if torch.cuda.is_available() else "cpu"

    if device == "cuda":
        print(f"device:{device}, cuDNN enabled: {torch.backends.cudnn.enabled}")

    with wandb.init(
            project=cfg.wandb.project,
            mode=cfg.wandb.mode,
            name=run_name,
            config=OmegaConf.to_container(cfg, resolve=True),
    ):

        func_build_dataset = get_func_build_dataset(cfg.model.func_build_dataset)
        train_loader, valid_loader, test_loader = func_build_dataset(cfg)

        model = build_model(cfg)
        model.to(device)

        for pipeline_name in cfg.model.pipeline:

            if cfg.model[pipeline_name].trainable != "all":
                for param in model.parameters():
                    param.requires_grad = False

                for layer_name in cfg.model[pipeline_name].trainable:
                    layer = model.get_submodule(layer_name)
                    for param in layer.parameters():
                        param.requires_grad = True
            else:
                for param in model.parameters():
                    param.requires_grad = True

            for name, param in model.named_parameters():
                print(name, param.requires_grad)

            optimizer = build_optimizer(model, cfg.model[pipeline_name].optimizer)
            scheduler = build_scheduler(optimizer, cfg.model[pipeline_name].scheduler)
            criterion = build_criterion(cfg.model[pipeline_name].criterion)

            if cfg.model[pipeline_name].early_stopping.enable:
                early_stopping = EarlyStopping(
                    mode="max",
                    warmup=cfg.model[pipeline_name].early_stopping.warmup,
                    patience=cfg.model[pipeline_name].early_stopping.patience,
                )
            else:
                early_stopping = None

            best_acc = 0
            save_name = f"{cfg.model.batch_size}_epochs_{cfg.model[pipeline_name].n_epochs}_batch_size_{cfg.model[pipeline_name].optimizer.kwargs.lr}_lr"
            best_model_path = (
                    Path(cfg.path.model)
                    / cfg.model.sname
                    / f"{cfg.model.sname}_{cfg.dataset.name}_{cfg.epoch_window.name}_{cfg.subject}_{save_name}.pt"
            )
            best_model_path.parent.mkdir(exist_ok=True, parents=True)

            for epoch in range(cfg.model[pipeline_name].n_epochs):
                t_start = time.time()
                train_loss, train_acc = train_one_epoch(
                    model,
                    train_loader,
                    optimizer,
                    criterion,
                    device=device,
                )
                valid_loss, valid_acc = validate(
                    model,
                    valid_loader,
                    criterion,
                    device=device,
                )
                duration = time.time() - t_start

                if cfg.model[pipeline_name].scheduler.need_loss:
                    scheduler.step(valid_loss)
                else:
                    scheduler.step()

                txt = f"{epoch + 1:03d}, train_loss: {train_loss:.4f}, train_acc: {train_acc:.4f}, valid_loss: {valid_loss:.4f}, valid_acc: {valid_acc:.4f}, duration: {duration:.2f}s, lr: {scheduler.get_last_lr()[0]:.4f}"

                if valid_acc > best_acc:
                    best_acc = valid_acc
                    torch.save(model.state_dict(), best_model_path)
                    txt += ", best model"

                print(txt)
                wandb.log(
                    {
                        "train_loss": train_loss,
                        "train_accuracy": train_acc,
                        "valid_loss": valid_loss,
                        "valid_accuracy": valid_acc,
                        "best_valid_accuracy": best_acc,
                        "time_one_epoch": duration,
                        "lr": scheduler.get_last_lr()[0],
                    }
                )

                if early_stopping is not None:
                    if early_stopping(valid_acc):
                        print("Early stopping")
                        break

            model.load_state_dict(torch.load(best_model_path, weights_only=True))
        test_loss, test_acc = validate(model, test_loader, criterion, device=device)
        wandb.log({"test_loss": test_loss, "test_accuracy": test_acc})

        # XAI analysis

        # delete some unused object to reduce GRAM
        del optimizer
        del scheduler
        del criterion
        del train_loader
        del valid_loader
        gc.collect()
        clear_cuda(model)

        test_loader_batch_1 = torch.utils.data.DataLoader(
            test_loader.dataset, batch_size=32, shuffle=False
        )

        attributions = {}
        attributions[0] = saliency_map(
            model, test_loader_batch_1, device, class_index=0
        )
        clear_cuda(model)
        attributions[1] = saliency_map(
            model, test_loader_batch_1, device, class_index=1
        )
        clear_cuda(model)

        attributions[0].update(
            integrated_gradients_map(model, test_loader_batch_1, device, class_index=0)
        )
        clear_cuda(model)
        attributions[1].update(
            integrated_gradients_map(model, test_loader_batch_1, device, class_index=1)
        )
        clear_cuda(model)

        attributions[0].update(
            deeplift_map(model, test_loader_batch_1, device, class_index=0)
        )
        clear_cuda(model)
        attributions[1].update(
            deeplift_map(model, test_loader_batch_1, device, class_index=1)
        )
        clear_cuda(model)

        attribution_base = Path(cfg.path.attributions)
        attribution_base.mkdir(exist_ok=True, parents=True)
        fname = (
                attribution_base
                / f"{cfg.model.sname}_{cfg.dataset.name}_{cfg.epoch_window.name}_{cfg.subject}.pt"
        )
        torch.save(attributions, fname)

        save_results(cfg, test_acc)
        print(
            f"Done: {cfg.model.sname} | {cfg.dataset.name} | {cfg.epoch_window.name} | subject {cfg.subject} → {test_acc:.4f}"
        )


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg):
    if cfg.debug:
        from src.utils import debug_warning

        debug_warning()

    if cfg.dataset.subjects.exclude is not None:
        if cfg.subject in cfg.dataset.subjects.exclude:
            print(f"subject: {cfg.subject} is excluded.")
            return

    print("CUBLAS_WORKSPACE_CONFIG =", os.environ.get("CUBLAS_WORKSPACE_CONFIG"))
    set_seed(cfg.seed)

    run(cfg)


if __name__ == "__main__":
    main()
