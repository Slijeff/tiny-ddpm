from typing import Dict, Optional, Tuple
from sympy import Ci
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from torchvision.datasets import CIFAR10
from torchvision import transforms
from torchvision.utils import save_image, make_grid

from unet import NaiveUnet
from ddpm import DDPM

import sys
sys.path.append("../../datasets")
from anime_girl_faces import AnimeGirlFacesDataset
from heshijie import Heshijie


def train(
    n_epoch: int = 100, device: str = "cuda", load_pth: Optional[str] = None, generate_every = 5, acc_grad = 4, img_size = 64
) -> None:

    ddpm = DDPM(eps_model=NaiveUnet(3, 3, n_feat=128), betas=(1e-4, 0.02), n_T=1000)

    if load_pth is not None:
        ddpm.load_state_dict(torch.load(load_pth))

    ddpm.to(device)

    # tf = transforms.Compose(
    #     [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    # )

    dataset = AnimeGirlFacesDataset(64)
    # dataset = Heshijie()

    dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=4, drop_last=True)
    optim = torch.optim.Adam(ddpm.parameters(), lr=3e-5)

    for i in range(n_epoch):
        print(f"Epoch {i} : ")
        ddpm.train()

        loss_ema = None
        pbar = tqdm(dataloader)
        for idx, x in enumerate(pbar):
            # optim.zero_grad()
            x = x.to(device)
            loss = ddpm(x)
            loss /= acc_grad
            loss.backward()
            if loss_ema is None:
                loss_ema = loss.item()
            else:
                loss_ema = 0.9 * loss_ema + 0.1 * loss.item()
            pbar.set_description(f"loss: {loss_ema:.4f}")
            # optim.step()

            if ((idx + 1) % acc_grad == 0) or (idx + 1 == len(pbar)):
                torch.nn.utils.clip_grad_norm_(ddpm.eps_model.parameters(), 1.0)
                optim.step()
                optim.zero_grad()

        ddpm.eval()
        if (i) % generate_every == 0 or i == n_epoch - 1:
            with torch.no_grad():
                xh = ddpm.sample(4, (3, img_size, img_size), device)
                xset = torch.cat([xh, x[:8]], dim=0)
                grid = make_grid(xset, normalize=True, value_range=(-1, 1), nrow=2)
                save_image(grid, f"./contents/ddpm_sample_{i + 1}.png")

            # save model
        torch.save(ddpm.state_dict(), f"./contents/ddpm.pth")


if __name__ == "__main__":
    train(n_epoch=500, generate_every=10, acc_grad=4, img_size=64, load_pth="./contents/ddpm.pth")
