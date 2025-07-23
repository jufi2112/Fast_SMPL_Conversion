# Copyright (c) 2025 Julien Fischer, Chair of Computer Graphics and Visualization, TUD Dresden University of Technology.
# All rights reserved.
# Licensed under the MIT License. See LICENSE in the project root for license information.

import torch

from torch import nn
from typing import Tuple, Dict, Union
from smpl_conversion.utils.model_infos import MODEL_STATS
from smpl_conversion.utils.parameter_processing import direct_transfer_pose_parameters
from smpl_conversion.utils.enum_configurations import PoseRepresentation, ConversionNetworkArchitecture, NetworkNormalizationMode, NetworkActivationFunction, BodyModelType


def assemble_combined_network_input(n_shape_components: int,
                                    input_dict: Dict = None,
                                    trans: torch.Tensor = None,
                                    betas: torch.Tensor = None,
                                    poses: torch.Tensor = None,
                                    ) -> torch.Tensor:
    """
        Assembles the given parameters to form a valid input to the CombinedConversion network

        Params
        ------
            n_shape_components (int):
                Number of shape components
            input_dict (Dict):
                A dictionary-like object that contains keys 'betas', 'poses', and 'trans' / 'transl'
                Defaults to None.
            trans (torch.Tensor):
                Tensor containing the translation parameters
            betas (torch.Tensor):
                Tensor containing the shape parameters
            poses (torch.Tensor):
                Tensor containing the pose parameters

        Returns
        -------
            torch.Tensor:
                A tensor that combines all input parameters and which can be directly given
                to the CombinedConversion network's forward method.
    """
    if input_dict is not None:
        trans = input_dict.get('trans', None)
        if trans is None:
            trans = input_dict.get('transl', None)
        betas = input_dict.get('betas', None)
        poses = input_dict.get('poses', None)

        if trans is None or betas is None or poses is None:
            raise ValueError(
                "The given dictionary did not contain one of the following "
                "keys: 'trans' (or 'transl'), 'betas', 'poses'. "
                f"Got {input_dict.keys()}"
            )
    inp = torch.cat((trans, betas[..., :n_shape_components], poses.reshape(poses.shape[0], -1)), dim=1)
    return inp



def split_network_output(network_output: torch.Tensor,
                         n_shape_components: int
                         ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
        Splits the given CombinedConversion network output in separate
        tensors for the respective parameters

        Params
        ------
            network_output (torch.Tensor):
                The CombinedConversion network output
            n_shape_components (int):
                The number of shape components that was used.

        Returns
        -------
            torch.tensor:
                Translation parameters
            torch.tensor:
                Shape parameters
            torch.tensor:
                Pose parameters
    """
    trans = network_output[:, :3]
    betas = network_output[:, 3:(n_shape_components+3)]
    poses = network_output[:, (n_shape_components+3):]
    return trans, betas, poses



class CombinedConversionNetwork (nn.Module):
    """
        A neural network that jointly predicts pose, shape, and translation parameter conversions
        from one specific parameterized human body model type to another.
    """
    def __init__(self,
                 model_from: BodyModelType,
                 model_to: BodyModelType,
                 num_shape_components: int,
                 architecture_type: Union[str, ConversionNetworkArchitecture],
                 max_dims: int,
                 network_normalization: Union[str, NetworkNormalizationMode],
                 network_activation: Union[str, NetworkActivationFunction],
                 input_rotation_representation: PoseRepresentation,
                 output_rotation_representation: PoseRepresentation
                 ):
        """
            A neural network that jointly predicts pose, shape, and translation parameter conversions
            from one specific parameterized human body model type to another.

            Params
            ------
                model_from (BodyModelType):
                    Model type whose parameters to convert.
                model_to (BodyModelType):
                    Model type to which the parameters should be converted.
                num_shape_components (int):
                    Number of shape components (used for both model types).
                architecture_type (str or ConversionNetworkArchitecture):
                    Architecture type to use. Supported are 'combined_single' or 'combined_per_parameter'.
                max_dims (int):
                    Maximum depth of the fully connected layers.
                network_normalization (str or NetworkNormalizationMode):
                    Type of normalization to apply between layers
                network_activation (str or NetworkActivationFunction):
                    Type of activation function that should be used.
                input_rotation_representation (PoseRepresentation):
                    Rotation representation of the input poses
                output_rotation_representation (PoseRepresentation):
                    Rotation representation of the output poses
        """
        super().__init__()
        self.model_from = model_from
        self.model_to = model_to
        self.n_shape_components = num_shape_components
        self.architecture_type = architecture_type
        self.input_rotation_representation = input_rotation_representation
        self.output_rotation_representation = output_rotation_representation
        if isinstance(self.architecture_type, str):
            self.architecture_type = ConversionNetworkArchitecture.from_string(self.architecture_type)
            if self.architecture_type is None:
                raise ValueError(
                    f"The requested network architecture {architecture_type} is unknown."
                )
        if not self.architecture_type.is_combined_architecture():
            raise ValueError(f"The provided architecture type {self.architecture_type} is not a combined network architecture.")
        self.max_dims = max_dims
        self.network_normalization = network_normalization
        if isinstance(self.network_normalization, str):
            self.network_normalization = NetworkNormalizationMode.from_string(self.network_normalization)
            if self.network_normalization is None:
                raise ValueError(
                    f"The requested network normalization mode {network_normalization} is unknown."
                )
        self.net_norm = self.network_normalization.to_string()
        self.norm_layer = self.network_normalization.get_pytorch_layer_class()
        self.activation_fct = network_activation
        if isinstance(self.activation_fct, str):
            self.activation_fct = NetworkActivationFunction.from_string(self.activation_fct)
            if self.activation_fct is None:
                raise ValueError(f"The requested network activation function {network_activation} is unknown.")
        self.act_fct = self.activation_fct.get_pytorch_layer_class()
        self.input_pose_size, self.output_pose_size = self._determine_input_output_size_pose()
        # 3... translation
        self.input_size = 3 + self.n_shape_components + self.input_pose_size
        self.output_size = 3 + self.n_shape_components + self.output_pose_size

        if self.architecture_type in [ConversionNetworkArchitecture.COMBINED_SINGLE,
                                      ConversionNetworkArchitecture.COMBINED_SKIP]:
            self.parameter_indices = None
            steps = [self.max_dims]
            dims = self.max_dims
            while dims // 2 > self.output_size:
                dims = dims // 2
                steps.append(dims)
            self.layers = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(self.input_size, self.max_dims),
                    self.act_fct(),
                    self.norm_layer(self.max_dims)
                )
            ])
            self.layers.extend([
                nn.Sequential(
                    nn.Linear(steps[i], steps[i + 1]),
                    self.act_fct(),
                    self.norm_layer(steps[i + 1])
                )
                if i < (len(steps) - 1)
                else nn.Sequential(
                    nn.Linear(steps[-1], self.output_size)
                )
                for i in range(len(steps))
            ])
        elif self.architecture_type == ConversionNetworkArchitecture.COMBINED_SKIP_LINEAR:
            self.network = nn.Linear(self.input_size, self.output_size)
        elif self.architecture_type == ConversionNetworkArchitecture.COMBINED_SKIP_WITHOUT_UPSAMPLING:
            self.network = nn.Sequential(
                nn.Linear(self.input_size, self.input_size),
                self.act_fct(),
                self.norm_layer(self.input_size),
                nn.Linear(self.input_size, self.output_size)
            )
        elif self.architecture_type == ConversionNetworkArchitecture.COMBINED_PER_PARAMETER:
            self.parameter_indices = {
                'trans': 0,
                'betas': 1,
                'poses': 2
            }
            self.layers = nn.ModuleList([
                nn.ModuleList([
                    nn.Sequential(
                        nn.Linear(self.input_size, self.max_dims),
                        self.act_fct(),
                        self.norm_layer(self.max_dims)
                    )
                ])
                for _ in self.parameter_indices.keys()
            ])
            for param, idx in self.parameter_indices.items():
                steps = [self.max_dims]
                dims = self.max_dims
                pred_dimension = self._get_parameter_dimensions(self.model_to,
                                                                self.output_rotation_representation,
                                                                param)
                while dims // 2 > pred_dimension:
                    dims = dims // 2
                    steps.append(dims)
                self.layers[idx].extend([
                    nn.Sequential(
                        nn.Linear(steps[i], steps[i + 1]),
                        self.act_fct(),
                        self.norm_layer(steps[i + 1])
                    )
                    if i < (len(steps) - 1)
                    else nn.Sequential(
                        nn.Linear(steps[-1], pred_dimension)
                    )
                    for i in range(len(steps))
                ])


    def _get_parameter_dimensions(self,
                                  model_type: BodyModelType,
                                  rot_repr: PoseRepresentation,
                                  param: str
                                  ) -> int:
        if param == 'trans':
            return 3
        elif param == 'betas':
            return self.n_shape_components
        elif param == 'poses':
            return MODEL_STATS[model_type].joints * rot_repr.get_number_components()


    def forward(self, x):
        if self.architecture_type in [ConversionNetworkArchitecture.COMBINED_SINGLE,
                                      ConversionNetworkArchitecture.COMBINED_SKIP]:
            if self.architecture_type == ConversionNetworkArchitecture.COMBINED_SKIP:
                inp_trans = x[:, :3]
                inp_shape = x[:, 3: (self.n_shape_components+3)]
                dt_pose = direct_transfer_pose_parameters(self.model_from,
                                                          self.model_to,
                                                          self.input_rotation_representation,
                                                          self.output_rotation_representation,
                                                          x[:, (self.n_shape_components + 3):])
            # Forward pass through all layers
            for i, l in enumerate(self.layers):
                logits = l(x) if i == 0 else l(logits)
            # Skip connections
            if self.architecture_type == ConversionNetworkArchitecture.COMBINED_SKIP:
                pred_trans = inp_trans + logits[:, :3]
                pred_shape = inp_shape + logits[:, 3: (self.n_shape_components + 3)]
                pred_pose = dt_pose + logits[:, (self.n_shape_components + 3):]
                logits = torch.hstack((pred_trans, pred_shape, pred_pose))
        elif self.architecture_type in [ConversionNetworkArchitecture.COMBINED_SKIP_LINEAR,
                                        ConversionNetworkArchitecture.COMBINED_SKIP_WITHOUT_UPSAMPLING]:
            inp_trans = x[:, :3]
            inp_shape = x[:, 3: (self.n_shape_components+3)]
            dt_pose = direct_transfer_pose_parameters(self.model_from,
                                                      self.model_to,
                                                      self.input_rotation_representation,
                                                      self.output_rotation_representation,
                                                      x[:, (self.n_shape_components+3):])
            logits = self.network(x)
            pred_trans = inp_trans + logits[: ,:3]
            pred_shape = inp_shape + logits[:, 3: (self.n_shape_components + 3)]
            pred_pose = dt_pose + logits[:, (self.n_shape_components + 3):]
            logits = torch.hstack((pred_trans, pred_shape, pred_pose))
        elif self.architecture_type == ConversionNetworkArchitecture.COMBINED_PER_PARAMETER:
            predictions = []
            for param, idx in self.parameter_indices.items():
                for i, l in enumerate(self.layers[idx]):
                    param_pred = l(x) if i == 0 else l(param_pred)
                predictions.append(param_pred)
            logits = torch.cat(predictions, dim=1)
        return logits


    def _split_input_parameters(self,
                                x: torch.Tensor
                                ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        trans = x[:, :3]
        betas = x[:, 3 : (self.n_shape_components + 3)]
        poses = x[:, (self.n_shape_components + 3):]
        return trans, betas, poses


    def _determine_input_output_size_pose(self) -> Tuple[int, int]:
        """
            Returns the number of pose parameters for the input and target model types.
            This is dependent on the rotation representation that the network should learn.

            Returns
            -------
                int:
                    input pose parameters
                int:
                    output pose parameters
        """
        return int(MODEL_STATS[self.model_from].joints * self.input_rotation_representation.get_number_components()), int(MODEL_STATS[self.model_to].joints * self.output_rotation_representation.get_number_components())
