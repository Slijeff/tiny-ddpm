from torchvision.utils import save_image, make_grid
from unet import NaiveUnet
from ddpm import DDPM
import torch

def generate():
    ddpm = DDPM(eps_model=NaiveUnet(3, 3, n_feat=128), betas=(1e-4, 0.02), n_T=1000)
    ddpm.load_state_dict(torch.load("./contents/ddpm.pth"))
    ddpm.to("cuda")

    with torch.no_grad():
        xh = ddpm.sample(8, (3, 64, 64), device="cuda")
        grid = make_grid(xh, normalize=True, value_range=(-1, 1), nrow=4)
        save_image(grid, f"./generated.png")

if __name__ == "__main__":
    generate()
