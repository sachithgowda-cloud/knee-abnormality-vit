import timm
import torch.nn as nn


def build_model(cfg):
    model = timm.create_model(
        cfg["model"]["name"],
        pretrained=cfg["model"]["pretrained"],
        num_classes=cfg["model"]["num_classes"],
        drop_rate=cfg["model"]["drop_rate"],
        img_size=cfg["model"]["img_size"],
    )
    return model
