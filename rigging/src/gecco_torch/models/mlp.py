from typing import Callable
import torch.nn as nn


class MLP(nn.Sequential):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        width_size: int,
        depth: int = 1,
        activation: Callable = nn.ReLU,
    ):
        layers = []
        layers.append(
            nn.Linear(
                in_features,
                width_size,
            )
        )
        layers.append(activation())

        for _ in range(depth - 1):
            layers.append(
                nn.Linear(
                    width_size,
                    width_size,
                )
            )
            layers.append(activation())

        layers.append(
            nn.Linear(
                width_size,
                out_features,
            )
        )

        super().__init__(*layers)
