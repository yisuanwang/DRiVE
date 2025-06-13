from typing import Any
import torch
import matplotlib.pyplot as plt
import lightning.pytorch as pl

from gecco_torch.diffusion import Diffusion
from gecco_torch.structs import Example


def plot_3d(clouds, colors=["blue", "red", "green"], shared_ax=False, images=None):
    assert len(clouds) <= len(colors)
    if images is not None:
        assert len(images) == len(clouds)
        assert not shared_ax

    width = 1 if shared_ax else len(clouds)
    height = 1 if images is None else 2
    figsize = (5 * width, 5 * height)
    fig = plt.figure(figsize=figsize, constrained_layout=True)

    def init_ax(ax):
        y_max = max(cloud[:, 1].max().detach().cpu().numpy() for cloud in clouds)
        y_min = min(cloud[:, 1].min().detach().cpu().numpy() for cloud in clouds)
        ax.view_init(azim=0, elev=0)
        ax.set_zlim(y_max, y_min)
        ax.set_aspect("equal")

    if shared_ax:
        ax = fig.add_subplot(projection="3d")
        init_ax(ax)

    for i, (cloud, color) in enumerate(zip(clouds, colors)):
        if not shared_ax:
            ax = fig.add_subplot(height, width, i + 1, projection="3d")
            init_ax(ax)

        x, y, z = cloud.detach().cpu().numpy().T
        ax.scatter(z, x, y, s=0.1, color=color)

    if images is not None:
        for i, image in enumerate(images):
            ax = fig.add_subplot(height, width, width + i + 1)
            ax.imshow(image.detach().cpu().numpy().transpose(1, 2, 0))
            ax.axis("off")

    return fig


class PCVisCallback(pl.Callback):
    """
    A callback which visualizes two things
        1. The context images (only once)
        2. The ground truth and sampled point clouds (once per validation phase)
    """

    def __init__(self, n: int = 8, n_steps: int = 64, point_size: int = 0.1):
        super().__init__()
        self.n = n
        self.n_steps = n_steps
        self.point_size = point_size
        self.batch: Example | None = None

    def on_validation_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: Diffusion,
        outputs: Any,
        batch: Example,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        if batch_idx != 0:
            return

        if self.batch is None:
            # cache the first batch for reproducible visualization
            # and display the context images (only once)
            self.batch = batch.apply_to_tensors(lambda t: t[: self.n].clone())

            if bool(self.batch.ctx):
                for i, image in enumerate(self.batch.ctx.image):
                    pl_module.logger.experiment.add_image(
                        tag=f"val/context_image_{i}",
                        img_tensor=image,
                        global_step=trainer.current_epoch,
                        dataformats="CHW",
                    )

        with torch.random.fork_rng(), torch.no_grad():
            torch.manual_seed(42)

            samples = pl_module.sample_stochastic(
                shape=self.batch.data.shape,
                context=self.batch.ctx,
                sigma_max=pl_module.sigma_max,
                num_steps=self.n_steps,
            )

        if not bool(self.batch.ctx):
            # no point showing "ground truth" for unconditional generation
            vertices = samples
            colors = None
        else:
            # concatenate context and samples for visualization
            # distinguish them by color
            vertices = torch.cat([self.batch.data, samples], dim=1)

            colors = torch.zeros(
                *vertices.shape, device=vertices.device, dtype=torch.uint8
            )
            colors[:, : self.batch.data.shape[1], 1] = 255  # green for ground truth
            colors[:, self.batch.data.shape[1] :, 0] = 255  # red for samples

        pl_module.logger.experiment.add_mesh(
            tag="val/samples",
            vertices=vertices,
            colors=colors,
            global_step=trainer.current_epoch,
            config_dict={
                "material": {
                    "cls": "PointsMaterial",
                    "size": self.point_size,
                },
            },
        )
