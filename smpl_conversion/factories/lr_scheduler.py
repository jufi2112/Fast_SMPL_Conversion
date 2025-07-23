# Copyright (c) 2025 Julien Fischer, Chair of Computer Graphics and Visualization, TUD Dresden University of Technology.
# All rights reserved.
# Licensed under the MIT License. See LICENSE in the project root for license information.

import torch
from typing import Dict, Union


class LRSchedulerFactory():
    """Scheduler names can also be given with underscores instead of CamelCase"""
    supported_schedulers = {
        'reducelronplateau': torch.optim.lr_scheduler.ReduceLROnPlateau,
        'cosineannealinglr': torch.optim.lr_scheduler.CosineAnnealingLR,
        'linearlr': torch.optim.lr_scheduler.LinearLR,
        'constantlr': torch.optim.lr_scheduler.ConstantLR
    }
    @staticmethod
    def create_lr_scheduler(scheduler_name: str,
                            scheduler_args: Dict
                            ) -> Union[torch.optim.lr_scheduler._LRScheduler, torch.optim.lr_scheduler.ReduceLROnPlateau]:
        scheduler_name = scheduler_name.lower()
        if scheduler_name not in LRSchedulerFactory.supported_schedulers.keys():
            # try to remove underscores
            scheduler_name_no_underscore = ''.join(c for c in scheduler_name if c != '_')
            if scheduler_name_no_underscore not in LRSchedulerFactory.supported_schedulers.keys():
                raise ValueError(
                    f"Unsupported scheduler {scheduler_name}. Expected one of "
                    f"{list(LRSchedulerFactory.supported_schedulers.keys())}"
                )
            else:
                scheduler_name = scheduler_name_no_underscore
        scheduler = LRSchedulerFactory.supported_schedulers[scheduler_name](
            **scheduler_args
        )
        return scheduler