from unet import NaiveUnet
from ddpm import DDPM
from torchsummary import summary

eps = NaiveUnet(3, 3, 128).to('cuda')
model = DDPM(eps_model=eps, betas=(0.0004, 0.02), n_T=1000).to('cuda')
print(summary(model, (4, 3,64,64)))


