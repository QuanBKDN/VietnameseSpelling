from dataclasses import dataclass

import yaml


@dataclass
class TrainConfig:
    pretrained_model: str
    data_path: str
    save_path: str
    num_epochs: int
    learning_rate: float
    batchsize: int
    weight_decay: float
    limit_save: int


@dataclass
class EvaluateConfig:
    data_path: str
    model_path: str
    log_path: str


@dataclass
class VietacConfig:
    train: TrainConfig


def read_config(config_path: str) -> VietacConfig:
    with open(config_path, "r") as stream:
        configs = yaml.safe_load(stream)
    train_config = TrainConfig(
        pretrained_model=configs["training"]["base_config"]["pretrained_model"],
        data_path=configs["training"]["base_config"]["data_path"],
        save_path=configs["training"]["base_config"]["save_path"],
        num_epochs=configs["training"]["parameters"]["num_epochs"],
        learning_rate=float(configs["training"]["parameters"]["learning_rate"]),
        batchsize=configs["training"]["parameters"]["batchsize"],
        weight_decay=configs["training"]["parameters"]["weight_decay"],
        limit_save=configs["training"]["parameters"]["limit_save"],
    )
    return VietacConfig(train=train_config)
