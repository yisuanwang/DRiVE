from typing import Any
from torch import nn, Tensor

from gecco_torch.models.set_transformer import SetTransformer


class LinearLift(nn.Module):
    """
    Embeds the 3d geometry (xyz points) in a higher dimensional space, passes it through
    the SetTransformer, and then maps it back to 3d. "Lift" refers to the embedding action.
    This class is used in the unconditional ShapeNet experiments.
    """

    def __init__(
        self,
        inner: SetTransformer,
        feature_dim: int,
        geometry_dim: int = 3,
        do_norm: bool = True,
    ):
        super().__init__()
        self.lift = nn.Linear(geometry_dim, feature_dim)
        self.inner = inner

        if do_norm:
            self.lower = nn.Sequential(
                nn.LayerNorm(feature_dim, elementwise_affine=False),
                nn.Linear(feature_dim, geometry_dim),
            )
        else:
            self.lower = nn.Linear(feature_dim, geometry_dim)

    def forward(
        self,
        geometry: Tensor,
        embed: Tensor,
        raw_context: Any,
        post_context: Any,
        do_cache: bool = False,
        cache: list[Tensor] | None = None,
    ) -> tuple[Tensor, list[Tensor] | None]:
        del raw_context, post_context  # unused

        features = self.lift(geometry)
        features, out_cache = self.inner(features, embed, do_cache, cache)
        return self.lower(features), out_cache
