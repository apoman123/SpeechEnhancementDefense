# Implementation of Consistency Model
# https://arxiv.org/pdf/2303.01469.pdf


from typing import List
from tqdm import tqdm
import math

import torch
from torch.utils.data import DataLoader

from torchvision.datasets import MNIST, CIFAR10
from torchvision import transforms
from torchvision.utils import save_image, make_grid
from torch.utils.tensorboard import SummaryWriter
from diffusers import UNet2DModel
import torch.nn.functional as F

from consistency_models import ConsistencyModel, kerras_boundaries


def mnist_dl():
    tf = transforms.Compose(
        [
            transforms.Pad(2),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5)),
        ]
    )

    dataset = MNIST(
        "./data",
        train=True,
        download=True,
        transform=tf,
    )

    dataloader = DataLoader(dataset, batch_size=128, shuffle=True, num_workers=20)

    return dataloader


def cifar10_dl():
    tf = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )

    dataset = CIFAR10(
        "./data",
        train=True,
        download=True,
        transform=tf,
    )

    dataloader = DataLoader(dataset, batch_size=128, shuffle=True, num_workers=20)

    return dataloader

def get_model():
    return UNet2DModel(
              sample_size=128,  # the target image resolution
              in_channels=1,
              out_channels=1
              act_fn="silu",
              add_attention=True,
              attention_head_dim=64,
              attn_norm_num_groups=32,
              block_out_channels=[
                192,
                384,
                576,
                768
              ],
              center_input_sample=False,
              class_embed_type=None,
              down_block_types=[
                "ResnetDownsampleBlock2D",
                "AttnDownBlock2D",
                "AttnDownBlock2D",
                "AttnDownBlock2D"
              ],
              downsample_padding=1,
              downsample_type="resnet",
              flip_sin_to_cos=True,
              freq_shift=0,
              layers_per_block=3,
              mid_block_scale_factor=1,
              norm_eps=1e-05,
              norm_num_groups=32,
              num_class_embeds=1000,
              resnet_time_scale_shift="scale_shift",
              time_embedding_type="positional",
              up_block_types=[
                "AttnUpBlock2D",
                "AttnUpBlock2D",
                "AttnUpBlock2D",
                "ResnetUpsampleBlock2D"
              ],
              upsample_type="resnet"
            )

def train(
    n_epoch: int = 100,
    device="cuda:0",
    dataloader=mnist_dl(),
    n_channels=1,
    name="mnist",
):
    # model = ConsistencyModel(n_channels, D=256)
    model = get_model()
    model.to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=1e-4)

    # Define \theta_{-}, which is EMA of the params
    # ema_model = ConsistencyModel(n_channels, D=256)
    ema_model = get_model()
    ema_model.to(device)
    ema_model.load_state_dict(model.state_dict())

    for epoch in range(1, n_epoch):
        N = math.ceil(math.sqrt((epoch * (150**2 - 4) / n_epoch) + 4) - 1) + 1
        boundaries = kerras_boundaries(7.0, 0.002, N, 80.0).to(device)

        pbar = tqdm(dataloader)
        loss_ema = None
        model.train()
        for x, _ in pbar:
            optim.zero_grad()
            x = x.to(device)

            z = torch.randn_like(x)
            t = torch.randint(0, N - 1, (x.shape[0], 1), device=device)
            t_0 = boundaries[t]
            t_1 = boundaries[t + 1]

            # loss = model.loss(x, z, t_0, t_1, ema_model=ema_model)
            
            loss.backward()
            if loss_ema is None:
                loss_ema = loss.item()
            else:
                loss_ema = 0.9 * loss_ema + 0.1 * loss.item()

            optim.step()
            with torch.no_grad():
                mu = math.exp(2 * math.log(0.95) / N)
                # update \theta_{-}
                for p, ema_p in zip(model.parameters(), ema_model.parameters()):
                    ema_p.mul_(mu).add_(p, alpha=1 - mu)

            pbar.set_description(f"loss: {loss_ema:.10f}, mu: {mu:.10f}")

        model.eval()
        with torch.no_grad():
            # Sample 5 Steps
            xh = model.sample(
                torch.randn_like(x).to(device=device) * 80.0,
                list(reversed([5.0, 10.0, 20.0, 40.0, 80.0])),
            )
            xh = (xh * 0.5 + 0.5).clamp(0, 1)
            grid = make_grid(xh, nrow=4)
            save_image(grid, f"./contents/ct_{name}_sample_5step_{epoch}.png")

            # Sample 2 Steps
            xh = model.sample(
                torch.randn_like(x).to(device=device) * 80.0,
                list(reversed([2.0, 80.0])),
            )
            xh = (xh * 0.5 + 0.5).clamp(0, 1)
            grid = make_grid(xh, nrow=4)
            save_image(grid, f"./contents/ct_{name}_sample_2step_{epoch}.png")

            # save model
            torch.save(model.state_dict(), f"./ct_{name}.pth")


if __name__ == "__main__":
    # train()
    train(dataloader=cifar10_dl(), n_channels=3, name="cifar10")
