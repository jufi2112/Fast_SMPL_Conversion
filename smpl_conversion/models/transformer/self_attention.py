# Copyright (c) 2025 Julien Fischer, Chair of Computer Graphics and Visualization, TUD Dresden University of Technology.
# All rights reserved.
# Licensed under the MIT License. See LICENSE in the project root for license information.

import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Union
from smpl_conversion.utils.enum_configurations import PositionalEncodingMode
from smpl_conversion.models.transformer.rotation_vector_embedding import RotationVectorEmbedding
from smpl_conversion.models.transformer.positional_encoding import PositionalEncoding


# class SelfAttention(nn.Module):
#     def __init__(self,
#                  embed_size: int,
#                  n_heads: int
#                  ):
#         """
#             Self attention block

#         Params
#         ------
#             embed_size (int):
#                 Embedding size
#             n_heads (int):
#                 Number of heads
#         """
#         super().__init__()
#         self.embed_size = embed_size
#         self.n_heads = n_heads
#         self.head_dim = embed_size // n_heads

#         assert self.head_dim * n_heads == embed_size, "Embedding size must be divisible by n_heads"
#         self.values = nn.Linear(self.head_dim, self.head_dim, bias=False)
#         self.keys = nn.Linear(self.head_dim, self.head_dim, bias=False)
#         self.queries = nn.Linear(self.head_dim, self.head_dim, bias=False)
#         self.fc_out = nn.Linear(n_heads*self.head_dim, embed_size)

class SingleHeadSelfAttention(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.query = nn.Linear(embed_dim, embed_dim, bias=False)
        self.key = nn.Linear(embed_dim, embed_dim, bias=False)
        self.value = nn.Linear(embed_dim, embed_dim, bias=False)
        self.register_buffer('scale', torch.sqrt(torch.FloatTensor([embed_dim])))


    def forward(self, x):
        """
            Self attention forward pass

            Returns
            -------
                Attention output of shape (B, 55, embed_size)
        """
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)

        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        attention_weights = F.softmax(attention_scores, dim=-1)
        attention_output = torch.matmul(attention_weights, V)
        
        return attention_output
