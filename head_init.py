import torch.nn as nn


def init_last_layer(layer: nn.Linear) -> None:
    nn.init.normal_(layer.weight, mean=0.0, std=0.01)
    nn.init.zeros_(layer.bias)
