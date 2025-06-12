import torch
import torch.nn as nn


class GaussianActivation(nn.Module):
    """
    Using the activation function proposed in
    "Beyond Periodicity: Towards a Unifying Framework for Activations in Coordinate-MLPs" by Ramasinghe et al.
    allows us to skip Fourier embedding low dimensional inputs such as noise level and 3D coordinates.
    """

    def __init__(self, normalized: bool = True):
        super().__init__()
        self.alpha = nn.Parameter(torch.tensor(1.0))
        self.normalized = normalized

    def forward(self, x):
        y = (-(x**2) / (2 * self.alpha**2)).exp()
        if self.normalized:
            # normalize by activation mean and std assuming
            # `x ~ N(0, 1)`
            y = (y - 0.7) / 0.28

        return y
