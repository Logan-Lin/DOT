import math

import torch
from torch import nn
import numpy as np
from einops import rearrange, repeat

from model.denoiser import Unet


class UNetPredictor(nn.Module):
    """
    Scalar predictor based on UNet.
    """

    def __init__(self, dim, init_channel=None, dim_mults=(1, 2, 4), in_channel=3,
                 resnet_block_groups=8, use_convnext=True, convnext_mult=2):
        super().__init__()

        self.unet = Unet(dim=dim, init_dim=init_channel, out_dim=1, 
                         dim_mults=dim_mults, channels=in_channel, with_time_emb=False,
                         resnet_block_groups=resnet_block_groups, use_convnext=use_convnext,
                         convnext_mult=convnext_mult)
        self.linear = nn.Sequential(nn.Linear(dim * dim, dim),
                                    nn.LeakyReLU(), nn.Dropout(0.1),
                                    nn.Linear(dim, 1))

        self.num_layers = len(dim_mults)
        self.d_model = init_channel
        self.use_st = True
        self.use_grid = True

        self.name = 'unet'

    def forward(self, x, odt):
        """
        :param x: input trajectory images, shape (N, num_channel, dim, dim)
        :param odt:
        :return:
        """
        x = self.unet(x).squeeze(1)  # (batch_size, dim, dim)
        x = x.reshape(x.size(0), -1)  # (batch_size, dim * dim)
        x = self.linear(x).squeeze(-1)   # (batch_size)
        return x


class PositionalEncoding(nn.Module):
    """
    A type of trigonometric encoding for indicating items' positions in sequences.
    """
    def __init__(self, embed_size, max_len):
        super().__init__()

        pe = torch.zeros(max_len, embed_size).float()
        pe.requires_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, embed_size, 2).float() * -(math.log(10000.0) / embed_size)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :x.size(1)]


class ContinuousEncoding(nn.Module):
    """
    A type of trigonometric encoding for encode continuous values into distance-sensitive vectors.
    """
    def __init__(self, embed_size):
        super().__init__()
        self.omega = nn.Parameter((torch.from_numpy(1 / 10 ** np.linspace(0, 9, embed_size))).float(),
                                  requires_grad=True)
        self.bias = nn.Parameter(torch.zeros(embed_size).float(), requires_grad=True)
        self.div_term = math.sqrt(1. / embed_size)

    def forward(self, x):
        """
        :param x: input sequence for encoding, (batch_size, seq_len)
        :return: encoded sequence, shape (batch_size, seq_len, embed_size)
        """
        encode = x.unsqueeze(-1) * self.omega.reshape(1, 1, -1) + self.bias.reshape(1, 1, -1)
        encode = torch.cos(encode)
        return self.div_term * encode


class TransformerPredictor(nn.Module):
    """
    Scalar predictor based on Transformer.
    """

    def __init__(self, input_dim, d_model, num_head, num_layers, num_grid, dropout, use_grid=True, use_st=True):
        super().__init__()

        self.num_grid = num_grid
        self.use_grid = use_grid
        self.use_st = use_st

        trans_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=num_head,
                                                 dim_feedforward=d_model, dropout=dropout, batch_first=True)
        self.transformer = nn.TransformerEncoder(trans_layer, num_layers=num_layers)

        self.input_linear = nn.Linear(input_dim - 1, d_model)
        self.grid_embed = nn.Embedding(num_grid, d_model)
        self.pos_encode = PositionalEncoding(d_model, num_grid)
        # self.val_encode = ContinuousEncoding(d_model)
        self.out_linear = nn.Linear(d_model, 1)

        self.num_layers = num_layers
        self.d_model = d_model

        self.name = 'trans'

    def forward(self, x, odt):
        """
        :param x: input trajectory images, shape (N, num_channel, dim, dim).
        :param odt:
        :return:
        """
        if len(x.shape) > 3:
            x = rearrange(x, 'b c h w -> b (h w) c')  # (N, num_grid, num_channel)
        else:
            x = rearrange(x, 'b c s -> b s c')
        # Create key padding mask according to the first channel of x.
        mask = x[:, :, 0] < 0  # (N, num_grid)
        pos = self.pos_encode(x)  # (N, num_grid, d_model)
        # pos = self.val_encode(x[:, :, -1])

        x = self.input_linear(x[:, :, 1:])  # (N, num_grid, d_model)
        grid = torch.arange(0, x.size(1)).long().to(x.device)
        grid = repeat(grid, 'g -> b g', b=x.size(0))  # (N, num_grid)
        grid = self.grid_embed(grid)  # (N, num_grid, d_model)

        if self.use_st: pos = pos + x
        if self.use_grid: pos = pos + grid
        out = self.transformer(pos, src_key_padding_mask=mask)  # (N, num_grid, d_model)
        out = self.out_linear(out).mean(1).squeeze(-1)
        out = torch.nan_to_num(out, nan=0.0)
        return out