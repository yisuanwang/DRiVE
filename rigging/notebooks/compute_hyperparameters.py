import torch
import gecco_torch
import matplotlib.pyplot as plt
import lightning.pytorch as pl
from einops import rearrange
from tqdm import tqdm
from torch import Tensor
from torch.utils.data import DataLoader


config = gecco_torch.load_config('../example_configs/train_joints.py')
data: pl.LightningDataModule = config.data
data.setup() # lightning data modules need to be setup before they can be used

batches = []

dataloader: DataLoader = data.train_dataloader()

for i, batch in tqdm(enumerate(dataloader)):
    # if i == 1000: # break early to save time
    #     break
    batches.append(batch)

batches: gecco_torch.structs.Example = dataloader.collate_fn(batches) # [5, batch_size, ...]
batches = batches.apply_to_tensors(lambda t: t.reshape(-1, *t.shape[2:])) # [5 * batch_size, ...]

placeholder_reparam = gecco_torch.reparam.GaussianReparam(
   mean=torch.zeros(3),
   sigma=torch.ones(3),
)

diff_placeholder = placeholder_reparam.data_to_diffusion(batches.data, batches.ctx)


def data_statistics(data: Tensor) -> tuple[Tensor, Tensor]:
    # x, y, z = data.unbind(-1)
    # fig, ax = plt.subplots()
    # kw = dict(histtype='step', bins=torch.linspace(data.min(), data.max(), 100))
    # ax.hist(x.flatten(), label=f'x mu={x.mean().item():.02f}, sigma={x.std().item():0.2f}', **kw)
    # ax.hist(y.flatten(), label=f'y mu={y.mean().item():.02f}, sigma={y.std().item():0.2f}', **kw)
    # ax.hist(z.flatten(), label=f'z mu={z.mean().item():.02f}, sigma={z.std().item():0.2f}', **kw)
    # fig.legend()

    mean = data.mean(dim=(0, 1))
    sigma = data.std(dim=(0, 1))

    return mean, sigma

mean_raw, sigma_raw = data_statistics(diff_placeholder)

adjusted_reparam = gecco_torch.reparam.GaussianReparam(mean=mean_raw, sigma=sigma_raw)
print(adjusted_reparam)

diff_adjusted = adjusted_reparam.data_to_diffusion(batches.data, batches.ctx)

data_statistics(diff_adjusted)