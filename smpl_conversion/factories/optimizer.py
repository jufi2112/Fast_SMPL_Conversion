# Copyright (c) 2025 Julien Fischer, Chair of Computer Graphics and Visualization, TUD Dresden University of Technology.
# All rights reserved.
# Licensed under the MIT License. See LICENSE in the project root for license information.

import torch
from typing import Dict

class OptimizerFactory():
    supported_optimizers = {
        'sgd': torch.optim.SGD,
        'adam': torch.optim.Adam
    }
    @staticmethod
    def create_optimizer(optim_name: str,
                         optim_args: Dict
                         ) -> torch.optim.Optimizer:
        optim_name = optim_name.lower()
        if optim_name not in OptimizerFactory.supported_optimizers.keys():
            raise ValueError(
                f"Unsupported optimizer: {optim_name}. Expected one of "
                f"{list(OptimizerFactory.supported_optimizers.keys())}"
            )
        optimizer = OptimizerFactory.supported_optimizers[optim_name](
            **optim_args
        )
        return optimizer