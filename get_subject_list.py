import hydra
from omegaconf import DictConfig


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):
    n_subjects = cfg.dataset.subjects.n_subjects
    subject_list = list(range(1, n_subjects + 1))
    exclude = cfg.dataset.subjects.exclude
    if exclude is not None:
        for subject in exclude:
            subject_list.remove(subject)

    for s in subject_list:
        print(s, end=" ")


if __name__ == "__main__":
    main()
