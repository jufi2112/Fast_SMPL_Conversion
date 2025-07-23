# Copyright (c) 2025 Julien Fischer, Chair of Computer Graphics and Visualization, TUD Dresden University of Technology.
# All rights reserved.
# Licensed under the MIT License. See LICENSE in the project root for license information.

import math
import torch
import torch.nn as nn

from typing import Union
from smpl_conversion.utils.enum_configurations import PositionalEncodingMode

class PositionalEncoding(nn.Module):
    def __init__(self,
                 embed_dim: int,
                 max_len: int,
                 pos_enc_mode: Union[str, PositionalEncodingMode]
                 ):
        """
            Positional encoding layer

            Params
            ------
                embed_dim (int):
                    Size of the embedding
                max_len (int):
                    Maximum number of vectors
                pos_enc_mode (PositionalEncodingMode):
                    Mode of the positional encoding.
        """
        super().__init__()
        self.mode = pos_enc_mode
        if isinstance(self.mode, str):
            self.mode = PositionalEncodingMode.from_string(self.mode)
            if self.mode is None:
                raise ValueError(f"Unrecognized positional encoding mode: {pos_enc_mode}")
        if self.mode == PositionalEncodingMode.LEARNABLE:
            self.learnable_encoding = nn.Parameter(torch.zeros(1, max_len, embed_dim))
            nn.init.normal_(self.learnable_encoding, mean=0, std=0.02)
        elif self.mode == PositionalEncodingMode.FIXED:
            pe = torch.zeros(max_len, embed_dim)
            position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
            div_term = torch.exp(torch.arange(0, embed_dim, 2).float() * (-math.log(10000.0) / embed_dim))
            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)
            pe = pe.unsqueeze(0)
            self.register_buffer('pe', pe)


    def forward(self, x):
        if self.mode == PositionalEncodingMode.LEARNABLE:
            x += self.learnable_encoding[:, :x.shape[1], :]
        elif self.mode == PositionalEncodingMode.FIXED:
            x += self.pe[:, :x.shape[1], :]
        return x
