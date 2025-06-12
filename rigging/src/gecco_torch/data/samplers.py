from dataclasses import dataclass
import torch
from torch import Tensor
from torch.utils.data import Dataset


@dataclass
class ConcatenatedSampler:
    data_length: int
    length: int
    rng: torch.Generator

    def __init__(
        self,
        data_source: Dataset,
        length: int,
        seed: int | None = None,
    ):
        self.data_length = len(data_source)
        self.length = length
        self.seed = seed

    def __len__(self):
        return self.length

    def __iter__(self):
        rng = torch.Generator()
        if self.seed is not None:
            seed = seed.manual_seed(self.seed)

        yielded = 0

        while yielded < len(self):
            permutation = torch.randperm(self.data_length, generator=rng)
            left_to_yield = len(self) - yielded
            yield from permutation[:left_to_yield].tolist()
            yielded += permutation.shape[0]


@dataclass
class FixedSampler:
    permutation: Tensor

    def __init__(
        self,
        dataset: Dataset,
        length: int | None = None,
        seed: int = 42,
    ):
        if length is None:
            length = len(dataset)
        if length > len(dataset):
            raise ValueError(f"{length=} is more than {len(dataset)=}.")

        rng: torch.Generator = torch.Generator().manual_seed(seed)
        perm = torch.randperm(len(dataset), generator=rng)
        self.permutation = perm[:length]

    def __len__(self):
        return self.permutation.shape[0]

    def __iter__(self):
        yield from self.permutation.tolist()
