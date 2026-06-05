import hydra
import wandb
from omegaconf import DictConfig, OmegaConf


def run(cfg_dict):
    print(cfg_dict)



@hydra.main(config_path="conf", config_name="config", version_base=None)
def main(cfg: DictConfig):
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    print(cfg_dict)

    run_name = f"{cfg.dataset.name}_{cfg.model.name}_{cfg.epoch_window.name}"

    run(cfg_dict)
    exit()

    wandb.init(
        project=cfg.wandb.project,
        mode=cfg.wandb.mode,
        name=run_name,
        config=cfg_dict,
    )

    print(OmegaConf.to_yaml(cfg))

    # ここで実際の学習
    # dataset = cfg.dataset.name
    # model_name = cfg.model.name
    # t0 = cfg.epoch_window.t0
    # t1 = cfg.epoch_window.t1

    wandb.log({"val_acc": 0.8})

    wandb.finish()


if __name__ == "__main__":
    main()
