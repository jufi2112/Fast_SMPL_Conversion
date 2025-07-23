# Copyright (c) 2025 Julien Fischer, Chair of Computer Graphics and Visualization, TUD Dresden University of Technology.
# All rights reserved.
# Licensed under the MIT License. See LICENSE in the project root for license information.

# Wrapper around PyTorch's TransformerEncoder with settings specific for the
# pose conversion transformer
# We're using PyTorch's TransformerEncoderLayer with norm_first=True and 
# batch_first=True


import torch
from torch import nn
import numpy as np
import os
from os import path as osp
from typing import Union, Optional
from smpl_conversion.models.transformer import SingleHeadSelfAttention
from smpl_conversion.utils.enum_configurations import PositionalEncodingMode



class TransformerEncoderBlock(nn.Module):
    def __init__(self,
                 embed_dim: int,
                 ffwd_hidden_size: int
                 ):
        """
            Encoder block of the transformer consisting of
            SingleHeadSelfAttention and Feed Forward Neural Network

            Params
            ------
                embed_dim (int):
                    Embedding size for each vector
                ffwd_hidden_size (int):
                    Size of the hidden layer in the feed forward fully connected network
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.ffwd_hidden_size = ffwd_hidden_size

        self.sa = SingleHeadSelfAttention(embed_dim)
        self.ln_1 = nn.LayerNorm(embed_dim)
        self.ffwd = nn.Sequential(nn.Linear(embed_dim, ffwd_hidden_size),
                                  nn.GELU(),
                                  nn.Linear(ffwd_hidden_size, embed_dim))
        self.ln_2 = nn.LayerNorm(embed_dim)


    def forward(self, x):
        """
            Forward pass of the transformer encoder

            Params
            ------
                x (torch.Tensor):
                    Input tensor of shape (B, J, embed_dim)

            Returns
            -------
                torch.Tensor:
                    Processed output of shape (B, J, embed_dim)
        """
        x = self.ln_1(x + self.sa(x))
        x = self.ln_2(x + self.ffwd(x))
        return x

# class TransformerEncoderWrapper(nn.Module):
#     def __init__(self,
#                  d_model: int,
#                  num_heads: int,
#                  dim_ffwd: int,
#                  num_layers: int = 6,
#                  dropout: float = 0.1,
#                  device: Optional[torch.device] = None
#                  ):
#         super().__init__()
#         self.enc_layer = nn.TransformerEncoderLayer(d_model,
#                                                     num_heads,
#                                                     dim_ffwd,
#                                                     dropout=dropout,
#                                                     batch_first=True,
#                                                     norm_first=True,
#                                                     device=device)
#         self.encoder = nn.TransformerEncoder(self.enc_layer,
#                                              num_layers)


#     def forward(self,
#                 x: torch.Tensor,
#                 mask: torch.Tensor = None
#                 ) -> torch.Tensor:
#         return self.encoder(x, mask=mask)
