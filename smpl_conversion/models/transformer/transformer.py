# Copyright (c) 2025 Julien Fischer, Chair of Computer Graphics and Visualization, TUD Dresden University of Technology.
# All rights reserved.
# Licensed under the MIT License. See LICENSE in the project root for license information.

# Wrapper around PyTorch's transformer model

import torch
import torch.nn as nn

from typing import Union
from smpl_conversion.utils.enum_configurations import PositionalEncodingMode
from smpl_conversion.models.transformer import PositionalEncoding, RotationVectorEmbedding, TransformerEncoderBlock


class PoseParameterTransformer(nn.Module):
    def __init__(self,
                 input_dim: int,
                 output_dim: int,
                 embed_dim: int,
                 input_seq_len: int,
                 output_seq_len: int,
                 pos_enc_mode: Union[str, PositionalEncodingMode],
                 ffwd_hidden_size: int,
                 n_blocks: int
                 ):
        """
            Transformer for the conversion of pose parameters

            Params
            ------
                input_dim (int):
                    Size of each input joint's pose representation
                output_dim (int):
                    Size of each output joint's pose representation
                embed_dim (int):
                    Embedding size for each pose representation
                input_seq_len (int):
                    Number of joints of the input body model type
                output_seq_len (int):
                    Number of joints of the target body model type
                pos_enc_mode (str or PositionalEncodingMode):
                    Determines whether positional encoding should be learnable
                    or fixed using sinusoidal encoding.
                ffwd_hidden_size (int):
                    Size of the hidden layer in the feed forward fully connected network
                n_blocks (int):
                    Number of encoder blocks to stack
        """
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.embed_dim = embed_dim
        self.input_seq_len = input_seq_len
        self.output_seq_len = output_seq_len
        self.pos_enc_mode = pos_enc_mode
        self.ffwd_hidden_size = ffwd_hidden_size
        self.n_blocks = n_blocks

        if isinstance(self.pos_enc_mode, str):
            self.pos_enc_mode = PositionalEncodingMode.from_string(self.pos_enc_mode)
            if self.pos_enc_mode is None:
                raise ValueError(f"Unrecognized option for positional encoding mode: {pos_enc_mode}")
        self.input_embedding = RotationVectorEmbedding(input_dim, embed_dim)
        self.position_encoding = PositionalEncoding(embed_dim, input_seq_len, self.pos_enc_mode)
        self.encoder_stack = nn.ModuleList(
            [
                TransformerEncoderBlock(embed_dim, ffwd_hidden_size) for _ in range(n_blocks)
            ]
        )
        self.output_dim_reduction = nn.Linear(embed_dim, output_dim)
        self.nl = nn.GELU()
        self.ln = nn.LayerNorm(output_dim)
        self.flatten = nn.Flatten(start_dim=1, end_dim=-1)
        self.final_fc = nn.Linear(input_seq_len * output_dim, output_seq_len * output_dim)


    def forward(self, x: torch.Tensor):
        """
            Forward pass

            Params
            ------
                x (torch.Tensor):
                    Input of shape (B, input_seq_len, input_dim)

            Returns
            -------
                torch.Tensor:
                    The transformer's output of shape (B, output_seq_len * output_dim)
        """
        x = self.input_embedding(x)                             # (B, input_seq_len, embed_dim)
        x = self.position_encoding(x)                           # (B, input_seq_len, embed_dim)
        for l in self.encoder_stack:
            x = l(x)                                            # (B, input_seq_len, embed_dim)
        x = self.ln(self.nl(self.output_dim_reduction(x)))      # (B, input_seq_len, output_dim)
        x = self.flatten(x)                                     # (B, input_seq_len * output_dim)
        x = self.final_fc(x)                                    # (B, output_seq_len * output_dim)
        return x

# class TransformerWrapper(nn.Module):
#     def __init__(self, config):
#         super().__init__()
#         self.device = config['training']['device']
#         if self.device == 'auto':
#             if torch.cuda.is_available():
#                 self.device = 'cuda'
#             else:
#                 self.device = 'cpu'
#         self.enc = TransformerEncoderWrapper(config['model']['d_model'],
#                                              config['model']['n_encoder_heads'],
#                                              config['model']['dim_ffwd'],
#                                              config['model']['n_encoder_layers'],
#                                              config['model']['dropout_encoder'],
#                                              self.device
#                                              )
#         self.dec = TransformerDecoderWrapper(config['model']['d_model'],
#                                              config['model']['n_decoder_heads'],
#                                              config['model']['dim_ffwd'],
#                                              config['model']['n_decoder_layers'],
#                                              config['model']['dropout_decoder'],
#                                              self.device
#                                              )
#         self.pos_enc = PositionalEncoding(config['model']['d_model'],
#                                           config['model']['dropout_pe']
#                                           )
#         self.linear = nn.Linear(config['model']['d_model'], config['model']['d_model'])
        


#     def forward(self,
#                 x: torch.Tensor,
#                 y: torch.Tensor
#                 ) -> torch.Tensor:
#         r"""Transformer Forward Call
#         Args:
#             x: Input sequence
#             y: Target sequence
#         Shape:
#             x: [B, J_inp, N] where B is batch size, J are the number of joints, and
#                 N is the dimensionality of the rotation representation.
#             y: [B, J_tgt, N]
#         """
#         dec_mask = get_decoder_input_mask()
#         x_pe = self.pos_enc(x)
#         y_pe = self.pos_enc(y)
#         enc_output = self.enc(x_pe)
#         dec_output = self.dec(dec_input=y_pe,
#                               enc_output=enc_output,
#                               dec_input_mask=dec_mask
#                               )
#         return self.linear(dec_output)
