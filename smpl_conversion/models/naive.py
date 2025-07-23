# Copyright (c) 2025 Julien Fischer, Chair of Computer Graphics and Visualization, TUD Dresden University of Technology.
# All rights reserved.
# Licensed under the MIT License. See LICENSE in the project root for license information.

import torch
import numpy as np

from torch import nn
from dotmap import DotMap
from typing import Tuple, Dict, Union
from smpl_conversion.utils.parameter_processing import direct_transfer_pose_parameters
from smpl_conversion.utils.enum_configurations import BodyModelType, PoseRepresentation

class NaiveConversionNetwork (nn.Module):
    """
        Naive conversion that only applies the direct transfer module to the input
    """
    def __init__(self,
                 model_from: BodyModelType,
                 model_to: BodyModelType,
                 num_shape_components: int,
                 input_rotation_representation: PoseRepresentation,
                 output_rotation_representation: PoseRepresentation
                 ):
        """
            Naive conversion that only applies the direct transfer module to the input

            Params
            ------
                model_from (BodyModelType):
                    Model type whose parameters to convert
                model_to (BodyModelType):
                    Model type to which the parameters should be converted.
                num_shape_components (int):
                    Number of shape components (used for both model types).
                input_rotation_representation (PoseRepresentation):
                    Rotation representation of the input poses
                output_rotation_representation (PoseRepresentation):
                    Rotation representation of the output poses
        """
        super().__init__()
        self.model_from = model_from
        self.model_to = model_to
        self.n_shape_components = num_shape_components
        self.input_rotation_representation = input_rotation_representation
        self.output_rotation_representation = output_rotation_representation


    def forward(self, x: Union[torch.Tensor, DotMap], split_output: bool = False
                ) -> Union[torch.Tensor, DotMap[str, torch.Tensor]]:
        """
            Params
            ------
                x (torch.Tensor or DotMap):
                    Input parameters of shape (B, N) with 
                    N = 3 + #betas + #poses model_from
                split_output (bool):
                    Whether output should be split. Defaults to False

            Returns
            -------
                torch.Tensor:
                    Directly (naively) transferred parameters of shape (B, M)
                    with M = 3 + #betas + #poses model_to
                DotMap:
                    Splitted translation, shape, and pose parameters
        """
        if isinstance(x, dict):
            inp_trans = x['trans']
            inp_shape = x['betas']
            inp_pose = x['poses']
        else:
            inp_trans = x[:, :3]
            inp_shape = x[:, 3:(self.n_shape_components+3)]
            inp_pose = x[:, (self.n_shape_components+3):]
        inp_trans = self._process_input(inp_trans)
        inp_shape = self._process_input(inp_shape)
        inp_pose = self._process_input(inp_pose)
        dt_pose = direct_transfer_pose_parameters(self.model_from,
                                                  self.model_to,
                                                  self.input_rotation_representation,
                                                  self.output_rotation_representation,
                                                  inp_pose)
        if split_output:
            conv_params = DotMap({
                'trans': inp_trans,
                'betas': inp_shape,
                'poses': dt_pose
            }, _dynamic=False)
        else:
            conv_params = torch.hstack((inp_trans, inp_shape, dt_pose))
        return conv_params


    def get_number_shape_components(self) -> int:
        return self.n_shape_components


    def _add_batch_dim(self, arr: Union[torch.Tensor, np.ndarray]
                       ) -> Union[torch.Tensor, np.ndarray]:
        if arr.ndim == 1:
            arr = arr[None, :]
        return arr


    def _process_input(self, arr: Union[torch.Tensor, np.ndarray]
                       ) -> torch.Tensor:
        if isinstance(arr, np.ndarray):
            arr = torch.from_numpy(arr)
        arr = self._add_batch_dim(arr)
        return arr
