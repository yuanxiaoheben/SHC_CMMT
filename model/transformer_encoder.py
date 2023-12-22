


import torch
import torch.nn as nn
import numpy as np
import math
from model.layers import PositionalEncoding, EncoderLayer





class Encoder(nn.Module):

    def __init__(self, configs):
        super().__init__()
        n_layers = configs.n_layers
        self.pos_emb = PositionalEncoding(configs.d_model, configs.max_pos_len, configs.device)

        self.layers = nn.ModuleList([EncoderLayer(d_model=configs.d_model,
                                                  ffn_hidden=configs.d_model,
                                                  n_head=configs.num_heads,
                                                  drop_prob=configs.drop_rate)
                                     for _ in range(n_layers)])

    def forward(self, x, s_mask=None):
        x = x + self.pos_emb(x)

        for layer in self.layers:
            x = layer(x, s_mask)

        return x