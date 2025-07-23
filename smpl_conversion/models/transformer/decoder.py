# Copyright (c) 2025 Julien Fischer, Chair of Computer Graphics and Visualization, TUD Dresden University of Technology.
# All rights reserved.
# Licensed under the MIT License. See LICENSE in the project root for license information.

# Wrapper around PyTorch's TransformerDecoder with settings specific for the
# pose conversion transformer
# We're using PyTorch's TransformerDecoderLayer with norm_first=True and
# batch_first=True

import torch
from torch import nn
import numpy as np
import os
from os import path as osp
from typing import Union, Optional

class TransformerDecoderWrapper(nn.Module):
    def __init__(self,
                 d_model: int,
                 num_heads: int,
                 dim_ffwd: int,
                 num_layers: int = 6,
                 dropout: float = 0.1,
                 device: Optional[torch.device] = None
                 ):
        super().__init__()
        self.dec_layer = nn.TransformerDecoderLayer(d_model,
                                                    num_heads,
                                                    dim_ffwd,
                                                    dropout=dropout,
                                                    batch_first=True,
                                                    norm_first=True,
                                                    device=device)
        self.decoder = nn.TransformerDecoder(self.dec_layer,
                                             num_layers)


    def forward(self,
                dec_input: torch.Tensor,
                enc_output: torch.Tensor,
                dec_input_mask: Optional[torch.Tensor] = None,
                enc_output_mask: Optional[torch.Tensor] = None
                ):
        return self.decoder(dec_input, enc_output, dec_input_mask, enc_output_mask)