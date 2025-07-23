# Copyright (c) 2025 Julien Fischer, Chair of Computer Graphics and Visualization, TUD Dresden University of Technology.
# All rights reserved.
# Licensed under the MIT License. See LICENSE in the project root for license information.

from .correspondence_loss import CorrespondenceLoss
from .direct_parameter_loss import DirectParameterLoss

__all__ = [CorrespondenceLoss, DirectParameterLoss]