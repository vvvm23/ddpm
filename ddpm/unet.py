import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange

import math
from typing import List
from copy import deepcopy

from .utils import HelperModule

# taken from:
# https://github.com/openai/improved-diffusion/blob/e94489283bb876ac1477d5dd7709bbbd2d9902ce/improved_diffusion/nn.py
# timesteps should be integers!!!!
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
class LayerNorm(nn.Module):
    def __init__(self, dim, eps = 1e-5):
        super().__init__()
        self.eps = eps
        self.g = nn.Parameter(torch.ones(1, dim, 1, 1))
        self.b = nn.Parameter(torch.zeros(1, dim, 1, 1))

    def forward(self, x):
        std = torch.var(x, dim = 1, unbiased = False, keepdim = True).sqrt()
        mean = torch.mean(x, dim = 1, keepdim = True)
        return (x - mean) / (std + self.eps) * self.g + self.b

# taken from:
# https://github.com/lucidrains/denoising-diffusion-pytorch/blob/master/denoising_diffusion_pytorch/denoising_diffusion_pytorch.py
class AttentionBlock(HelperModule):
    def build(self, dim, heads = 4, dim_head = 32):
        self.heads = heads
        hidden_dim = dim_head * heads
        self.pre_norm = LayerNorm(dim)
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias = False)
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x):
        b, c, h, w = x.shape
        x = self.pre_norm(x)
        qkv = self.to_qkv(x)
        q, k, v = rearrange(qkv, 'b (qkv heads c) h w -> qkv b heads c (h w)', heads = self.heads, qkv=3)
        k = k.softmax(dim=-1)
        context = torch.einsum('bhdn,bhen->bhde', k, v)
        out = torch.einsum('bhde,bhdn->bhen', context, q)
        out = rearrange(out, 'b heads c (h w) -> b (heads c) h w', heads=self.heads, h=h, w=w)
        return self.to_out(out) + x

class Block(HelperModule):
    def build(self,
            dim_in: int,
            dim_out: int,
            groups: int = 8,
        ):
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.layers = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, 3, padding=1),
            nn.GroupNorm(groups, dim_out),
            nn.SiLU(),
        )

    def forward(self, x, emb=None):
        return self.layers(x + (0.0 if emb == None else emb))

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
                nn.Linear(dim_emb, 2*dim_out),
                nn.SiLU()
            )
        else:
            self.block_out = Block(dim_out, dim_out, groups=groups)
            self.emb = nn.Sequential(
                nn.Linear(dim_emb, dim_out),
                nn.SiLU()
            )
        self.res_block = nn.Conv2d(dim_in, dim_out, 1) if dim_in != dim_out else nn.Identity()

    def forward(self, x, t_emb):
        h = self.block_in(x)
        t_emb = self.emb(t_emb)[:, :, None, None] # TODO: cast type?
        h = self.block_out(h, t_emb)
        return self.res_block(x) + h

class UNetSampleType:
    DOWN = 'down'
    UP = 'up'
    SAME = 'same'

class SampleLayer(HelperModule):
    def build(self,
        dim: int,
        sample_type: UNetSampleType,
    ):
        self.sample_type = sample_type
        if sample_type == UNetSampleType.DOWN:
            self.layer = nn.Conv2d(dim, dim, 3, 2, 1)
        elif sample_type == UNetSampleType.UP:
            self.layer = nn.ConvTranspose2d(dim, dim, 4, 2, 1)
        elif sample_type == UNetSampleType.SAME:
            self.layer = nn.Identity()
        else:
            raise ValueError(f"invalid sampling strategy '{sample_type}'")

    def forward(self, x):
        return self.layer(x)

class UNetLayer(HelperModule):
    def build(self,
            dim_in: int,
            dim_out: int,
            dim_emb: int,
            sample_type: UNetSampleType,
            nb_res_layers: int = 3,
            attention: bool = False,
            nb_heads: int = 4,
            dim_head: int = 32,
            scale_norm: bool = False,
            groups: int = 8,
        ):
        res_dims = [dim_in] + [dim_out] * (nb_res_layers - 1)
        
        self.blocks = nn.ModuleList([
            ResnetBlock(di, do, dim_emb, scale_norm=scale_norm, groups=groups)
            for di, do in zip(res_dims[:-1], res_dims[1:])
        ])

        # TODO: add pre_norm (LayerNorm) before attention block
        self.attn = AttentionBlock(dim_out, nb_heads, dim_head) if attention else nn.Identity()
        self.sample = SampleLayer(dim_out, sample_type)

    def forward(self, x, t):
        for res in self.blocks:
            x = res(x, t)
        h = self.attn(x)
        x = self.sample(h)
        return x, h

class UNet(HelperModule):
    # TODO: add dropout
    def build(self,
            in_channels: int = 3,
            base_channels: int = 128,
            channel_multipliers: List[int] = [1, 2, 2, 2],
            dim_t: int = 64, # TODO: sane?
            attention_resolutions: List[int] = [2, 4],
            nb_res_layers: int = 3,
            nb_heads: int = 4,
            dim_head: int = 32,
            scale_norm: bool = False,
            norm_groups: int = 8,
        ):

        self.dim_t = dim_t
        self.time_mlp = nn.Sequential(
            nn.Linear(dim_t, dim_t*4),
            nn.SiLU(),
            nn.Linear(dim_t*4, dim_t),
        )

        cumulative_sample = 1
        channels = [in_channels] + [base_channels * cm for cm in channel_multipliers]
        last_idx = len(channels) - 2

        self.down_layers = nn.ModuleList()
        for i, (ci, co) in enumerate(zip(channels[:-1], channels[1:])):
            self.down_layers.append(
                UNetLayer(
                    dim_in = ci, 
                    dim_out = co,
                    dim_emb = dim_t,
                    sample_type = UNetSampleType.SAME if i == last_idx else UNetSampleType.DOWN,
                    nb_res_layers = nb_res_layers,
                    attention = cumulative_sample in attention_resolutions,
                    nb_heads = nb_heads,
                    dim_head = dim_head,
                    scale_norm = scale_norm,
                    groups = norm_groups,
                )
            )
            cumulative_sample *= 1 if i == last_idx else 2

        self.mid_block = UNetLayer(
            dim_in = channels[-1], 
            dim_out = channels[-1],
            dim_emb = dim_t,
            sample_type = UNetSampleType.SAME,
            nb_res_layers = nb_res_layers,
            attention = cumulative_sample in attention_resolutions,
            nb_heads = nb_heads,
            dim_head = dim_head,
            scale_norm = scale_norm,
            groups = norm_groups,
        )

        self.up_layers = nn.ModuleList()
        channels = channels[1:]
        for i, (ci, co) in enumerate(zip(channels[:0:-1], channels[-2::-1])):
            self.up_layers.append(
                UNetLayer(
                    dim_in = ci*2, 
                    dim_out = co,
                    dim_emb = dim_t,
                    sample_type = UNetSampleType.SAME if i == last_idx else UNetSampleType.UP,
                    nb_res_layers = nb_res_layers,
                    attention = cumulative_sample in attention_resolutions,
                    nb_heads = nb_heads,
                    dim_head = dim_head,
                    scale_norm = scale_norm,
                    groups = norm_groups,
                )
            )
            cumulative_sample //= 1 if i == last_idx else 2

        self.out_block = nn.Sequential(
            # Block(in_channels, in_channels, groups=norm_groups),
            nn.Conv2d(channels[0], in_channels, 1),
        )

    def forward(self, x, t):
        t_emb = self.time_mlp(timestep_embedding(t, self.dim_t))

        activations = []
        for l in self.down_layers:
            x, h = l(x, t_emb)
            activations.append(h)

        x, _ = self.mid_block(x, t_emb)

        for l in self.up_layers:
            x, _ = l(torch.cat([x, activations.pop()], dim=1), t_emb)
        
        return self.out_block(x)

# TODO: start ema point
class EMA(HelperModule):
    def build(self,
            model: nn.Module,
            beta: float = 0.995,
        ):
        self.beta = beta
        self.model = model
        self.model_copy = deepcopy(self.model) # TODO: may be better as a buffer

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def update_average(self, old, new):
        return old * self.beta + new * (1. - self.beta)

    def update_model(self):
        for old_param, new_param in zip(self.model_copy.parameters(), self.model.parameters()):
            old_data, new_data = old_param.data, new_param.data
            self.model_copy.data = self.update_average(old_data, new_data)

    def reset_model(self):
        self.model_copy.load_state_dict(self.model.state_dict())

if __name__ == '__main__':
    from .utils import get_parameter_count
    N = 8
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = EMA(UNet(
        channel_multipliers = [1, 2, 2, 2],
        attention_resolutions = [16],
        scale_norm=False
    ), beta=0.995).to(device)
    print(f"> number of parameters: {get_parameter_count(model)}")
    x = torch.randn(N, 3, 256, 256).to(device)
    t = torch.randint(0, 5000, (N,)).to(device)
    y = model(x, t)
    print(y.shape)

    model.update_model()
    print(f"> updating model. beta = {model.beta}")
