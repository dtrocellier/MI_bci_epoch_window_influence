import time
from pathlib import Path

import hydra
import torch
from omegaconf import OmegaConf

import wandb
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
    compute_attributions,
)


def run(cfg):
    print(cfg)

    run_name = (
        f"{cfg.dataset.name}_{cfg.model.name}_{cfg.epoch_window.name}_sub-{cfg.subject}"
    )

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

            for name, param in model.named_parameters():
                print(name, param.requires_grad)

            optimizer = build_optimizer(model, cfg.model[pipeline_name].optimizer)
            scheduler = build_scheduler(optimizer, cfg.model[pipeline_name].scheduler)
            criterion = build_criterion(cfg.model[pipeline_name].criterion)

            best_acc = 0
            save_name = f"{cfg.model.batch_size}_epochs_{cfg.model[pipeline_name].n_epochs}_batch_size_{cfg.model[pipeline_name].optimizer.kwargs.lr}_lr"
            best_model_path = (
                Path("model")
                / cfg.model.name
                / f"{cfg.model.name}_{cfg.dataset.name}_{cfg.epoch_window.name}_{cfg.subject}_{save_name}.pt"
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

                if valid_acc > best_acc:
                    best_acc = valid_acc
                    torch.save(model.state_dict(), best_model_path)

                print(
                    f"{epoch + 1:03d}, train_loss: {train_loss:.4f}, train_acc: {train_acc:.4f}, valid_loss: {valid_loss:.4f}, valid_acc: {valid_acc:.4f}, duration: {duration:.2f}s, lr: {scheduler.get_last_lr()[0]:.4f}"
                )
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

            model.load_state_dict(torch.load(best_model_path, weights_only=True))
        test_loss, test_acc = validate(model, test_loader, criterion, device=device)
        wandb.log({"test_loss": test_loss, "test_accuracy": test_acc})

        attributions = {}
        attributions[0] = compute_attributions(
            model, test_loader, cfg, target=0, device=device
        )
        attributions[1] = compute_attributions(
            model, test_loader, cfg, target=1, device=device
        )

        # attribution_base = Path("attribution")
        attribution_base = Path(
            "/media/skojima/41e27c66-4999-42a0-b36f-cc19d7881326/david/attribution"
        )
        attribution_base.mkdir(exist_ok=True, parents=True)
        fname = (
            attribution_base
            / f"{cfg.model.name}_{cfg.dataset.name}_{cfg.epoch_window.name}_{cfg.subject}.pt"
        )
        torch.save(attributions, fname)

        save_results(cfg, test_acc)
        print(
            f"Done: {cfg.model.name} | {cfg.dataset.name} | {cfg.epoch_window.name} | subject {cfg.subject} → {test_acc:.4f}"
        )


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg):
    if cfg.subject in cfg.dataset.subjects.exclude:
        print(f"subject: {cfg.subject} is excluded.")
        return

    set_seed(cfg.seed)
    run(cfg)


if __name__ == "__main__":
    main()
