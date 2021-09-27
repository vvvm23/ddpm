import torch
import torch.nn.functional as F
import numpy as np

from math import pi

from .utils import HelperModule

class NoiseMode:
    LINEAR = 'linear'
    COSINE = 'cosine'

class Diffuser(HelperModule):
    def build(self,
            nb_timesteps: int   = 1000,
            mode: NoiseMode     = NoiseMode.COSINE,
            s: float            = 8e-3,
        ):
        self.nb_timesteps = nb_timesteps
        self.mode = mode
        self.s = s

        ts = torch.linspace(0.0, 1.0, nb_timesteps, dtype=torch.float64)

        if mode == NoiseMode.LINEAR:
            cum_alpha = torch.cumprod(1.0 - ts, dim=0)
        elif mode == NoiseMode.COSINE:
            ft = torch.cos((pi / 2) * (ts+s) / (1+s)).pow(2)
            cum_alpha = ft / ft[0]
        sqrt_cum_alpha = torch.sqrt(cum_alpha)
        one_minus_cum_alpha = 1.0 - cum_alpha

        self.register_buffer('sqrt_cum_alpha', sqrt_cum_alpha.to(torch.float32)[:, None, None, None])
        self.register_buffer('one_minus_cum_alpha', one_minus_cum_alpha.to(torch.float32)[:, None, None, None])

    def diffuse(self, x, t, noise):
        return self.sqrt_cum_alpha[t] * x + self.one_minus_cum_alpha[t] * noise

if __name__ == '__main__':
    from torchvision.utils import save_image
    import sys
    import imageio
    from tqdm import tqdm

    device = torch.device('cuda')
    def load_images(files):
        return torch.cat([torch.from_numpy(imageio.imread(f)).permute(-1, 0, 1).unsqueeze(0).long().to(device) for f in files], dim=0)

    T = 100
    diffuser = Diffuser(
        nb_timesteps=T,
        mode=NoiseMode.COSINE,
    ).to(device)
        
    x = load_images(sys.argv[1:]) / 255.
    x = F.interpolate(x, scale_factor=0.25)
    for i, t in enumerate(tqdm(torch.arange(T, dtype=torch.long))):
        y = diffuser.diffuse(x, t)
        save_image(y.cpu(), f"test-{str(i).zfill(5)}.jpg", nrow=2)
