# Copyright (c) 2025 Julien Fischer, Chair of Computer Graphics and Visualization, TUD Dresden University of Technology.
# All rights reserved.
# Licensed under the MIT License. See LICENSE in the project root for license information.

import torch
import torch.nn as nn

class RotationVectorEmbedding(nn.Module):
    def __init__(self, input_dim=3, embed_dim=64):
        """
            Embedding layer for rotation vectors.

            Params
            ------
                input_dim (int):
                    Components of the input rotation vector. Defaults to 3
                embed_dim (int):
                    Embedding dimension. Defaults to 64
        """
        super().__init__()
        self.embedding = nn.Linear(input_dim, embed_dim)


    def forward(self, x: torch.Tensor):
        return self.embedding(x)
