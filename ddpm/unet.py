import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange

import math
from typing import List

from .utils import HelperModule

# taken from:
# https://github.com/openai/improved-diffusion/blob/e94489283bb876ac1477d5dd7709bbbd2d9902ce/improved_diffusion/nn.py
# timesteps should be integers!
def timestep_embedding(timesteps, dim, max_period=10000):
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
    ).to(device=timesteps.device)
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding


# taken from:
# https://github.com/lucidrains/denoising-diffusion-pytorch/blob/master/denoising_diffusion_pytorch/denoising_diffusion_pytorch.py
class AttentionBlock(HelperModule):
    def __init__(self, dim, heads = 4, dim_head = 32):
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias = False)
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x)
        q, k, v = rearrange(qkv, 'b (qkv heads c) h w -> qkv b heads c (h w)', heads = self.heads, qkv=3)
        k = k.softmax(dim=-1)
        context = torch.einsum('bhdn,bhen->bhde', k, v)
        out = torch.einsum('bhde,bhdn->bhen', context, q)
        out = rearrange(out, 'b heads c (h w) -> b (heads c) h w', heads=self.heads, h=h, w=w)
        return self.to_out(out)

class Block(HelperModule):
    def build(self,
            dim_in: int,
            dim_out: int,
            groups: int = 8,
        ):
        self.layers = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, 3, padding=1),
            nn.GroupNorm(groups, dim_out),
            nn.SiLU(),
        )

    def forward(self, x, emb):
        return self.layers(x + emb)

class ScaleBlock(HelperModule):
    def build(self,
            dim_in: int,
            dim_out: int,
            groups: int = 8,
        ):
        self.layers = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, 3, padding=1),
            nn.GroupNorm(groups, dim_out)
        )

    def forward(self, x, emb):
        scale, shift = torch.chunk(emb, 2, dim=1)
        return F.silu(self.layers(x) * (1 + scale) + shift)

class ResnetBlock(HelperModule):
    def build(self,
            dim_in: int,
            dim_out: int,
            dim_emb: int,
            scale_norm: bool = False,
            groups: int = 8,
        ):
        self.scale_norm = scale_norm

        self.block_in = Block(dim_in, dim_out, groups=groups)
        if scale_norm:
            self.block_out = ScaleBlock(dim_out, dim_out, groups=groups)
            self.emb = nn.Sequential(
                nn.Linear(dim_out, 2*dim_out),
                nn.SiLU()
            )
        else:
            self.block_out = Block(dim_out, dim_out, groups=groups)
            self.emb = nn.Sequential(
                nn.Linear(dim_out, dim_out),
                nn.SiLU()
            )
        self.res_block = nn.Conv(dim_in, dim_out, 1) if dim_in != dim_out else nn.Identity()

    def forward(self, x, t_emb):
        h = self.block_in(x)
        t_emb = self.emb(t_emb)[:, :, None, None] # TODO: cast type?
        h = self.block_out(h, t_emb)
        return self.res_block(x) + h

class UNetLayer(HelperModule):
    def build(self,
            dim_in: int,
            dim_out: int,
            dim_emb: int,
            downsample: bool = True, #2x downsampling
            attention: bool = False,
            nb_heads: int = 4,
            dim_head: int = 32,
            scale_norm: bool = False,
            groups: int = 8,
        ):
        pass

    def forward(self, x):
        pass

class UNet(HelperModule):
    # TODO: add dropout
    def build(self,
            in_channels: int = 3,
            base_channels: int = 128,
            channel_multipliers: List[int] = [1, 2, 2, 2],
            attention_resolutions: List[int] = [2, 4],
            nb_res_blocks: int = 3,
            nb_heads: int = 4,
            dim_head: int = 32,
            norm_groups: int = 8,
        ):
        pass

    def forward(self, x):
        pass
