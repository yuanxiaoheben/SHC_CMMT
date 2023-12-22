


import torch
import torch.nn as nn
import numpy as np
import math
from model.layers import PositionalEncoding, DecoderLayer



class Decoder(nn.Module):
    def __init__(self, configs):
        super().__init__()
        n_layers = configs.n_layers
        self.pos_emb = PositionalEncoding(configs.d_model, configs.max_pos_len, configs.device)

        self.layers = nn.ModuleList([DecoderLayer(d_model=configs.d_model,
                                                  ffn_hidden=configs.d_model,
                                                  n_head=configs.num_heads,
                                                  drop_prob=configs.drop_rate)
                                     for _ in range(n_layers)])

        self.linear = nn.Linear(configs.d_model, configs.d_model)

    def forward(self, trg):
        trg = trg + self.pos_emb(trg)

        for layer in self.layers:
            trg = layer(trg)

        # pass to LM head
        output = self.linear(trg)
        return output