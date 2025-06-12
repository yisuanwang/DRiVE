import os

import torch
import lightning.pytorch as pl

from gecco_torch.diffusion import EDMPrecond, Diffusion
from gecco_torch.reparam import UVLReparam, GaussianReparam
from gecco_torch.diffusion import Diffusion, LogUniformSchedule, EDMLoss
from gecco_torch.models.set_transformer import SetTransformer
from gecco_torch.models.feature_pyramid import ConvNeXtExtractor
from gecco_torch.models.ray import RayNetwork
from gecco_torch.models.activation import GaussianActivation
from gecco_torch.vis import PCVisCallback
# from gecco_torch.data.skeleton_cond import TaskonomyDataModule
# from gecco_torch.data.skeleton_cond_mesh import TaskonomyDataModule
from gecco_torch.data.skeleton_test import TaskonomyDataModule
from gecco_torch.ema import EMACallback


train_dataset_path = "/oss/mingzesun/data/cpfs01/vrm/test_3dgs_demo/"
test_dataset_path = "/oss/mingzesun/data/cpfs01/vrm/test_3dgs_demo/"
rgb_dataset_path = "/oss/mingzesun/data/cpfs01/vrm/rendered_images_4view/"


NUM_STEPS = 2_000_000
SAVE_EVERY = 10_000
BATCH_SIZE = 29 # 48, 64, 16, 36
epoch_size = None
num_epochs = NUM_STEPS // SAVE_EVERY
print(
    f"num steps: {NUM_STEPS}, batch size: {BATCH_SIZE}, save_every: {SAVE_EVERY}, epoch size: {epoch_size}, num epochs: {num_epochs}"
)

# Reparametrization makes the point cloud more isotropic.
# The values below are computed with the notebook in notebooks/compute_hyperparams.ipynb
reparam = GaussianReparam(
    mean=torch.tensor([-0.0006006215699017048, 0.08998294919729233, 0.017616869881749153]),
    sigma=torch.tensor([0.17428095638751984, 0.28580644726753235, 0.049740005284547806]),
)

# Set up the network. RayNetwork is responsible for augmenting cloud features with local
# features extracted from the context 2d image.
network = RayNetwork(
    backbone=SetTransformer(  # a point cloud network with extra "global" input for the `t` parameter
        n_layers=6,
        num_inducers=64,
        feature_dim=3 * 128, #3 * 128
        t_embed_dim=1,  # dimensionality of the `t` parameter
        num_heads=8,
        activation=GaussianActivation,
    ),
    reparam=reparam,  # we need the reparam object to go between data and diffusion spaces for ray lookup
    context_dims=(96, 192, 384),  # feature dimensions at scale 1, 2, 4
)

# Set up the diffusion model. This is largely agnostic of the
# 3d point cloud task and could be used for 2d image diffusion as well.
model = Diffusion(
    backbone=EDMPrecond(
        model=network,
    ),
    conditioner=ConvNeXtExtractor(),
    reparam=reparam,
    loss=EDMLoss(
        schedule=LogUniformSchedule(
            max=180.0,
        ),
    ),
)

# set up the data module
data = TaskonomyDataModule(
    train_dataset_path,
    test_dataset_path,
    rgb_dataset_path,
    n_points=100,
    batch_size=BATCH_SIZE,
    num_workers=16,
    epoch_size=epoch_size,
    val_size=29, #184, 36
    input_type='v',
)


def trainer():
    return pl.Trainer(
        default_root_dir=os.path.split(__file__)[0],
        callbacks=[
            EMACallback(decay=0.999),
            pl.callbacks.ModelCheckpoint(),
            pl.callbacks.ModelCheckpoint(
                monitor="val_loss",
                filename="{epoch}-{val_loss:.3f}",
                save_top_k=1,
                mode="min",
            ),
            PCVisCallback(
                n=8, n_steps=128, point_size=0.05
            ),
        ],
        max_epochs=num_epochs,
        precision="16-mixed",
        gradient_clip_val=1.0,
        gradient_clip_algorithm="norm",
        strategy='ddp_find_unused_parameters_true', 
    )


if __name__ == "__main__":
    trainer().fit(model, data)
