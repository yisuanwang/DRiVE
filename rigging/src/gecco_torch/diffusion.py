"""
Definitions of the diffusion model itself, along with preconditioning, loss and sampling functions.
"""

from __future__ import annotations

import math
from typing import Any, Sequence
import torch
import lightning.pytorch as pl
from torch import nn, Tensor
from tqdm.auto import tqdm

from gecco_torch.reparam import Reparam, NoReparam
from gecco_torch.structs import Example, Context3d


def ones(n: int):
    return (1,) * n


class EDMPrecond(nn.Module):
    """
    Preconditioning module wrapping a diffusion backbone. Implements the logic proposed in
    "Elucidating the Design Space of Diffusion-Based Generative Models" by Kerras et al.
    """

    def __init__(
        self,
        model: nn.Module,
        sigma_data=1.0,
    ):
        super().__init__()
        self.model = model
        self.sigma_data = sigma_data

    def forward(
        self,
        x: Tensor,
        sigma: Tensor,
        raw_context: Any,  # raw_context comes from the dataset, before any preprocessing
        post_context: Any,  # post_context comes from the conditioner
        do_cache: bool = False,  # whether to return a cache of inducer states for upsampling
        cache: list[Tensor] | None = None,  # cache of inducer states for upsampling
    ) -> tuple[Tensor, list[Tensor] | None]:  # denoised, optional cache
        sigma = sigma.reshape(-1, *ones(x.ndim - 1))

        c_skip = self.sigma_data**2 / (sigma**2 + self.sigma_data**2)
        c_out = sigma * self.sigma_data / (sigma**2 + self.sigma_data**2).sqrt()
        c_in = 1 / (self.sigma_data**2 + sigma**2).sqrt()
        c_noise = sigma.log() / 4
        c_noise = c_noise

        F_x, cache = self.model(
            (c_in * x), c_noise, raw_context, post_context, do_cache, cache
        )
        denoised = c_skip * x + c_out * F_x

        if not do_cache:
            return denoised
        else:
            return denoised, cache


class LogNormalSchedule(nn.Module):
    """
    LogNormal noise schedule as proposed in "Elucidating the Design Space of Diffusion-Based Generative Models" by Kerras et al.
    """

    def __init__(self, sigma_max: float, mean=-1.2, std=1.2):
        super().__init__()

        self.sigma_max = sigma_max
        self.mean = mean
        self.std = std

    def extra_repr(self) -> str:
        return f"sigma_max={self.sigma_max}, mean={self.mean}, std={self.std}"

    def forward(self, data: Tensor) -> Tensor:
        rnd_normal = torch.randn(
            [data.shape[0], *ones(data.ndim - 1)], device=data.device
        )
        return (rnd_normal * self.P_std + self.P_mean).exp()


class LogUniformSchedule(nn.Module):
    """
    LogUniform noise schedule which seems to work better in our (GECCO) context.
    """

    def __init__(self, max: float, min: float = 0.002, low_discrepancy: bool = True):
        super().__init__()

        self.sigma_min = min
        self.sigma_max = max
        self.log_sigma_min = math.log(min)
        self.log_sigma_max = math.log(max)
        self.low_discrepancy = low_discrepancy

    def extra_repr(self) -> str:
        return f"sigma_min={self.sigma_min}, sigma_max={self.sigma_max}, low_discrepancy={self.low_discrepancy}"

    def forward(self, data: Tensor) -> Tensor:
        u = torch.rand(data.shape[0], device=data.device)

        if self.low_discrepancy:
            div = 1 / data.shape[0]
            u = div * u
            u = u + div * torch.arange(data.shape[0], device=data.device)

        sigma = (
            u * (self.log_sigma_max - self.log_sigma_min) + self.log_sigma_min
        ).exp()
        return sigma.reshape(-1, *ones(data.ndim - 1))


class EDMLoss(nn.Module):
    """
    A loss function for training diffusion models. Implements the loss proposed in
    "Elucidating the Design Space of Diffusion-Based Generative Models" by Kerras et al.
    """

    def __init__(
        self, schedule: nn.Module, sigma_data: float = 1.0, loss_scale: float = 100.0
    ):
        super().__init__()

        self.schedule = schedule
        self.sigma_data = sigma_data
        self.loss_scale = loss_scale

    def extra_repr(self) -> str:
        return f"sigma_data={self.sigma_data}, loss_scale={self.loss_scale}"

    def forward(self, net: Diffusion, examples: Example, context: Context3d) -> Tensor:
        ex_diff = net.reparam.data_to_diffusion(examples, context)
        sigma = self.schedule(ex_diff)
        weight = (sigma**2 + self.sigma_data**2) / ((sigma * self.sigma_data) ** 2)
        n = torch.randn_like(ex_diff) * sigma
        D_yn = net(ex_diff + n, sigma, context)
        loss = self.loss_scale * weight * ((D_yn - ex_diff) ** 2)
        return loss.mean()


class Conditioner(nn.Module):
    """
    An abstract class for a conditioner. Conditioners are used to preprocess the context
    before it is passed to the diffusion backbone.

    NOT TO BE CONFUSED with preconditioning the diffusion model itself (done by EDMPrecond).
    """

    def forward(self, raw_context):
        raise NotImplementedError()


class IdleConditioner(Conditioner):
    """
    A placeholder conditioner that does nothing, for unconditional models.
    """

    def forward(self, raw_context: Context3d | None) -> None:
        del raw_context
        return None


class Diffusion(pl.LightningModule):
    """
    The main diffusion model. It consists of a backbone, a conditioner, a loss function
    and a reparameterization scheme.

    It derives from PyTorch Lightning's LightningModule, so it can be trained with PyTorch Lightning trainers.
    """

    def __init__(
        self,
        backbone: nn.Module,
        conditioner: Conditioner,
        loss: EDMLoss,
        reparam: Reparam = NoReparam(dim=3),
    ):
        super().__init__()

        self.backbone = backbone
        self.conditioner = conditioner
        self.loss = loss
        self.reparam = reparam

        # set default sampler kwargs
        self.sampler_kwargs = dict(
            num_steps=64,
            sigma_min=0.002,
            sigma_max=self.sigma_max,
            rho=7,
            S_churn=0.5,
            S_min=0,
            S_max=float("inf"),
            S_noise=1,
            with_pbar=False,
        )

    def extra_repr(self) -> str:
        return str(self.sampler_kwargs)

    @property
    def sigma_max(self) -> float:
        return self.loss.schedule.sigma_max

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-4)

    def training_step(self, batch: Example, batch_idx):
        x, ctx = batch
        loss = self.loss(
            self,
            x,
            ctx,
        )
        self.log("train_loss", loss)

        return loss

    def validation_step(self, batch: Example, batch_idx):
        x, ctx = batch
        loss = self.loss(
            self,
            x,
            ctx,
        )
        self.log("val_loss", loss)

    def forward(
        self,
        data: Tensor,
        sigma: Tensor,
        raw_context: Any | None,
        post_context: Any | None = None,
        do_cache: bool = False,
        cache: Any | None = None,
    ) -> Tensor:
        """
        Applies the denoising network to the given data, with the given noise level and context.
        """
        
        if post_context is None:
            post_context = self.conditioner(raw_context)
        
        return self.backbone(data, sigma, raw_context, post_context, do_cache, cache)

    @property
    def example_param(self) -> Tensor:
        return next(self.parameters())

    def t_steps(
        self, num_steps: int, sigma_max: float, sigma_min: float, rho: float
    ) -> Tensor:
        """
        Returns an array of sampling time steps for the given parameters.
        """
        step_indices = torch.arange(
            num_steps, dtype=torch.float64, device=self.example_param.device
        )
        t_steps = (
            sigma_max ** (1 / rho)
            + step_indices
            / (num_steps - 1)
            * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))
        ) ** rho
        t_steps = torch.cat([t_steps, torch.zeros_like(t_steps[:1])])  # t_N = 0
        return t_steps

    @torch.no_grad()
    def sample_stochastic(
        self,
        shape: Sequence[int],
        context: Context3d | None,
        rng: torch.Generator = None,
        **kwargs,
    ) -> Tensor:
        """
        A stochastic sampling function that samples from the diffusion model with the given context. Corresponds to
        the `SDE` sampler in the paper. The `ODE` sampler is not currently implemented in PyTorch.
        """
        kwargs = {**self.sampler_kwargs, **kwargs}
        num_steps = kwargs["num_steps"]
        sigma_min = kwargs["sigma_min"]
        sigma_max = kwargs["sigma_max"]
        rho = kwargs["rho"]
        S_churn = kwargs["S_churn"]
        S_min = kwargs["S_min"]
        S_max = kwargs["S_max"]
        S_noise = kwargs["S_noise"]
        with_pbar = kwargs["with_pbar"]
        
        
        device = self.example_param.device
        dtype = self.example_param.dtype
        if rng is None:
            rng = torch.Generator(device).manual_seed(42)

        B = shape[0]
        latents = torch.randn(shape, device=device, generator=rng, dtype=dtype)

        post_context = self.conditioner(context)

        # Time step discretization.
        t_steps = self.t_steps(num_steps, sigma_max, sigma_min, rho)

        # Main sampling loop.
        x_next = latents.to(torch.float64) * t_steps[0]
        ts = list(enumerate(zip(t_steps[:-1], t_steps[1:])))

        if with_pbar:
            ts = tqdm(ts, unit="step")

        for i, (t_cur, t_next) in ts:  # 0, ..., N-1
            x_cur = x_next

            # Increase noise temporarily.
            gamma = (
                min(S_churn / num_steps, math.sqrt(2.0) - 1)
                if S_min <= t_cur <= S_max
                else 0
            )
            t_hat = t_cur + gamma * t_cur
            noise = torch.randn(x_cur.shape, device=device, generator=rng, dtype=dtype)
            x_hat = x_cur + (t_hat**2 - t_cur**2).sqrt() * S_noise * noise

            # Euler step.
            denoised = self(
                x_hat.to(dtype),
                t_hat.repeat(B).to(dtype),
                context,
                post_context,
            ).to(torch.float64)

            d_cur = (x_hat - denoised) / t_hat
            x_next = x_hat + (t_next - t_hat) * d_cur

            # Apply 2nd order correction.
            if i < num_steps - 1:
                denoised = self(
                    x_next.to(dtype),
                    t_next.repeat(B).to(dtype),
                    context,
                    post_context,
                ).to(torch.float64)
                d_prime = (x_next - denoised) / t_next
                x_next = x_hat + (t_next - t_hat) * (0.5 * d_cur + 0.5 * d_prime)

        if with_pbar:
            ts.close()

        return self.reparam.diffusion_to_data(x_next, context)

    @torch.no_grad()
    def upsample(
        self,
        data: Tensor,
        new_latents: Tensor | None = None,
        n_new: int | None = None,
        context: Context3d | None = None,
        seed: int | None = 42,
        num_substeps=5,
        **kwargs,
    ):
        """
        An upsampling function that upsamples the given data to the given number of new points.

        Args:
            `data` - the data to be upsampled
            `new_latents` or `n_new` - either the specific latent variables to use for upsampling
                if the user wants to control them, or the number of new points to generate.
            `context` - the context to condition the upsampling with

        Returns:
            The newly sampled points.
        """
        kwargs = {**self.sampler_kwargs, **kwargs}
        num_steps = kwargs["num_steps"]
        sigma_min = kwargs["sigma_min"]
        sigma_max = kwargs["sigma_max"]
        rho = kwargs["rho"]
        S_churn = kwargs["S_churn"]
        S_min = kwargs["S_min"]
        S_max = kwargs["S_max"]
        S_noise = kwargs["S_noise"]
        with_pbar = kwargs["with_pbar"]

        net_dtype: torch.dtype = self.example_param.dtype
        net_device: torch.device = self.example_param.device

        rng: torch.Generator = torch.Generator(device=net_device)
        if seed is not None:
            rng: torch.Generator = rng.manual_seed(seed)
        randn = lambda shape: torch.randn(
            shape, device=net_device, dtype=net_dtype, generator=rng
        )

        if (new_latents is None) == (n_new is None):
            raise ValueError(
                "Either new_latents or n_new must be specified, but not both."
            )
        if new_latents is None:
            new_latents = randn((data.shape[0], n_new, data.shape[2]))
        assert isinstance(new_latents, Tensor)

        data = self.reparam.data_to_diffusion(data, context)

        # Time step discretization.
        t_steps = self.t_steps(num_steps, sigma_max, sigma_min, rho)

        def call_net_cached(x: Tensor, t: Tensor, ctx: Context3d, cache: list[Tensor]):
            """
            A helper function that calls the diffusion backbone with the given inputs and cache.
            """
            return self(
                x.to(net_dtype),
                t.to(net_dtype).expand(x.shape[0]),
                ctx,
                do_cache=False,
                cache=cache,
            ).to(torch.float64)

        x_next = new_latents.to(torch.float64) * t_steps[0]

        steps = enumerate(zip(t_steps[:-1], t_steps[1:]))
        if with_pbar:
            steps = tqdm(steps, total=t_steps.shape[0] - 1)
        # Main sampling loop.
        for i, (t_cur, t_next) in steps:  # 0, ..., N-1
            data_ctx = data + randn(data.shape) * t_cur
            _, cache = self(
                data_ctx.to(net_dtype),
                t_cur.to(net_dtype).expand(data_ctx.shape[0]),
                context,
                do_cache=True,
                cache=None,
            )
            for u in range(num_substeps):
                x_cur = x_next

                # Increase noise temporarily.
                gamma = (
                    min(S_churn / num_steps, math.sqrt(2) - 1)
                    if S_min <= t_cur <= S_max
                    else 0
                )
                t_hat = t_cur + gamma * t_cur
                x_hat = x_cur + (t_hat**2 - t_cur**2).sqrt() * S_noise * randn(
                    x_cur.shape
                )

                # Euler step.
                denoised = call_net_cached(x_hat, t_hat, context, cache)
                d_cur = (x_hat - denoised) / t_hat
                x_next = x_hat + (t_next - t_hat) * d_cur

                # Apply 2nd order correction.
                if i < num_steps - 1:
                    denoised = call_net_cached(x_next, t_next, context, cache)
                    d_prime = (x_next - denoised) / t_next
                    x_next = x_hat + (t_next - t_hat) * (0.5 * d_cur + 0.5 * d_prime)

                if u < num_substeps - 1 and i < num_steps - 1:
                    redo_noise = (t_cur**2 - t_next**2).sqrt()
                    x_next = x_next + redo_noise * randn(x_next.shape)

        if with_pbar:
            steps.close()

        return self.reparam.diffusion_to_data(x_next, context)
