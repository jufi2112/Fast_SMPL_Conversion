# Copyright (c) 2025 Julien Fischer, Chair of Computer Graphics and Visualization, TUD Dresden University of Technology.
# All rights reserved.
# Licensed under the MIT License. See LICENSE in the project root for license information.

import torch
import torch.nn as nn

from dotmap import DotMap
from typing import Union, List

class DirectParameterLoss(nn.Module):
    def __init__(self):
        super().__init__()


    def forward(self,
                X: DotMap,
                y: DotMap,
                param_names: Union[List[str], str],
                l: int = 2,
                reduction_mode: str = 'mean', # or 'sum'
                loss_weights: Union[torch.tensor, List[float]] = None,
                **kwargs
                ) -> torch.tensor:
        if not isinstance(param_names, list):
            param_names = [param_names]
        if isinstance(loss_weights, list):
            loss_weights = torch.tensor(loss_weights, dtype=torch.float32)
        if loss_weights is not None and len(param_names) != len(loss_weights):
            raise ValueError(f"Expected param_names and loss_weights to have the same length, got {len(param_names)} vs {len(loss_weights)}")
        if loss_weights is None:
            # If not given, use equal weight for all
            loss_weights = torch.ones((len(param_names),), dtype=torch.float32) / len(param_names)
        overall_loss = torch.zeros((1,), dtype=torch.float32, requires_grad=True, device=X[param_names[0]].device)
        for idx, param in enumerate(param_names):
            loss = X[param] - y[param]
            if param == 'betas':
                # weight shape components according to order of appearance, e.g. first component has the heighest weight
                weights = torch.arange(loss.shape[1]+1)[1:].flip(dims=[-1]).to(loss.device)
                loss = loss * weights
            if l == 1:
                loss = loss.abs().sum(dim=-1)
            elif l == 2:
                loss = loss.pow(2).sum(dim=-1).sqrt()
            else:
                raise ValueError(f"DirectParameterLoss not implemented for argument l > 2, got {l}")
            if reduction_mode == 'sum':
                loss = loss.sum()
            loss = loss.mean()
            overall_loss = overall_loss + loss * loss_weights[idx].to(loss.device)
        return overall_loss
