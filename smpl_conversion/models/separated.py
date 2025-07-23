# Copyright (c) 2025 Julien Fischer, Chair of Computer Graphics and Visualization, TUD Dresden University of Technology.
# All rights reserved.
# Licensed under the MIT License. See LICENSE in the project root for license information.

import torch

from torch import nn
from typing import Tuple, Union, Dict
from smpl_conversion.utils.model_infos import MODEL_STATS
from smpl_conversion.models.transformer import PoseParameterTransformer
from smpl_conversion.utils.parameter_processing import direct_transfer_pose_parameters
from smpl_conversion.utils.enum_configurations import PoseRepresentation, ConversionNetworkArchitecture, PositionalEncodingMode, NetworkNormalizationMode, NetworkActivationFunction, BodyModelType


class SeparatedConversionNetwork (nn.Module):
    def __init__(self,
                 model_from: BodyModelType,
                 model_to: BodyModelType,
                 num_shape_components: int,
                 architecture_type: Union[str, ConversionNetworkArchitecture],
                 pose_max_dims: int,
                 network_normalization: Union[str, NetworkNormalizationMode],
                 network_activation: Union[str, NetworkActivationFunction],
                 input_rotation_representation: PoseRepresentation,
                 output_rotation_representation: PoseRepresentation,
                 attention_config: Dict = None
                 ):
        """
            A conversion network that separately predicts translation, shape, and
            pose parameters from one specific parameterized human body model to
            another.

            Params
            ------
                model_from (BodyModelType):
                    Model type who's parameters should be converted
                model_to (BodyModelType):
                    Model type to which the parameters should be converted
                num_shape_components (int):
                    Number of shape components. Same for both body model types
                architecture_type (str):
                    Type of the model architecture. Supported is 'separated_simple'
                pose_max_dims (int):
                    Maximum number of dimensions for the pose conversion subnetwork
                network_normalization (str or NetworkNormalizationMode)
                    Normalization mode that should be applied between layers.
                    Does not influence type of normalization in the transformer/
                    self attention block.
                network_activation (str or NetworkActivationFunction):
                    Type of activation function that should be used.
                input_rotation_representation (PoseRepresentation):
                    Input rotation representation
                output_rotation_representation (PoseRepresentation):
                    Output rotation representation
                attention_config (Dict):
                    If attention is utilized, the configuration for it
        """
        super().__init__()
        self.model_from = model_from
        self.model_to = model_to
        self.num_shapes = num_shape_components
        self.architecture_type = architecture_type
        self.input_rotation_representation = input_rotation_representation
        self.output_rotation_representation = output_rotation_representation
        if isinstance(self.architecture_type, str):
            self.architecture_type = ConversionNetworkArchitecture.from_string(self.architecture_type)
            if self.architecture_type is None:
                raise ValueError(f"Unsupported architecture type: {self.architecture_type}")
        if not self.architecture_type.is_separated_architecture():
            raise ValueError(f"The provided architecture type {self.architecture_type} is not a separated network architecture.")
        self.pose_max_dims = pose_max_dims
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
        if self.architecture_type in [ConversionNetworkArchitecture.SEPARATED_SIMPLE,
                                      ConversionNetworkArchitecture.SEPARATED_SKIP]:
            self.trans_network = nn.Sequential(nn.Linear(3, 3),
                                               self.act_fct(),
                                               self.norm_layer(3),
                                               nn.Linear(3,3))
            
            self.shape_network = nn.Sequential(nn.Linear(self.num_shapes, self.num_shapes),
                                               self.act_fct(),
                                               self.norm_layer(self.num_shapes),
                                               nn.Linear(self.num_shapes,self.num_shapes))
            steps = [self.pose_max_dims]
            dims = self.pose_max_dims
            while dims // 2 > self.output_pose_size:
                dims = dims // 2
                steps.append(dims)
            self.pose_network = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(self.input_pose_size, self.pose_max_dims),
                    self.act_fct(),
                    self.norm_layer(self.pose_max_dims)
                )
            ])
            self.pose_network.extend([
                nn.Sequential(
                    nn.Linear(steps[i], steps[i+1]),
                    self.act_fct(),
                    self.norm_layer(steps[i+1])
                )
                if i < (len(steps) - 1)
                else nn.Sequential(nn.Linear(steps[-1], self.output_pose_size))
                for i in range(len(steps))
            ])
        elif self.architecture_type in [ConversionNetworkArchitecture.SEPARATED_ATTENTION,
                                        ConversionNetworkArchitecture.SEPARATED_ATTENTION_SKIP]:
            self.trans_network = nn.Sequential(nn.Linear(3,3),
                                               self.act_fct(),
                                               self.norm_layer(3),
                                               nn.Linear(3,3))
            self.shape_network = nn.Sequential(nn.Linear(self.num_shapes, self.num_shapes),
                                               self.act_fct(),
                                               self.norm_layer(self.num_shapes),
                                               nn.Linear(self.num_shapes, self.num_shapes))
            self.transformer = PoseParameterTransformer(input_rotation_representation.get_number_components(),
                                                        output_rotation_representation.get_number_components(),
                                                        attention_config['embed_dim'],
                                                        MODEL_STATS[self.model_from].joints,
                                                        MODEL_STATS[self.model_to].joints,
                                                        attention_config['pos_enc_mode'],
                                                        attention_config['ffwd_hidden_size'],
                                                        attention_config['n_blocks'])


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        inp_trans = x[:, :3]
        inp_shape = x[:, 3: 3+self.num_shapes]
        inp_pose = x[:, 3+self.num_shapes:]
        if self.architecture_type == ConversionNetworkArchitecture.SEPARATED_SIMPLE:
            pred_trans = self.trans_network(inp_trans)
            pred_shape = self.shape_network(inp_shape)
            pred_pose = inp_pose
            for l in self.pose_network:
                pred_pose = l(pred_pose)
            return torch.hstack((pred_trans, pred_shape, pred_pose))
        elif self.architecture_type == ConversionNetworkArchitecture.SEPARATED_SKIP:
            pred_trans = inp_trans + self.trans_network(inp_trans)
            pred_shape = inp_shape + self.shape_network(inp_shape)
            pred_pose = inp_pose
            dt_pose = direct_transfer_pose_parameters(self.model_from,
                                                      self.model_to,
                                                      self.input_rotation_representation,
                                                      self.output_rotation_representation,
                                                      inp_pose)
            for l in self.pose_network:
                pred_pose = l(pred_pose)
            pred_pose = dt_pose + pred_pose
            return torch.hstack((pred_trans, pred_shape, pred_pose))
        elif self.architecture_type in [ConversionNetworkArchitecture.SEPARATED_ATTENTION,
                                        ConversionNetworkArchitecture.SEPARATED_ATTENTION_SKIP]:
            pred_trans = inp_trans + self.trans_network(inp_trans)
            pred_shape = inp_shape + self.shape_network(inp_shape)
            if self.architecture_type == ConversionNetworkArchitecture.SEPARATED_ATTENTION_SKIP:
                dt_pose = direct_transfer_pose_parameters(self.model_from,
                                                          self.model_to,
                                                          self.input_rotation_representation,
                                                          self.output_rotation_representation,
                                                          inp_pose)
            inp_pose = inp_pose.reshape(-1, MODEL_STATS[self.model_from].joints, self.input_rotation_representation.get_number_components())    # (N, input_joints, input_rot_repr)
            pred_pose = self.transformer(inp_pose)                                                                                              # (N, output_joints * output_rot_repr)
            if self.architecture_type == ConversionNetworkArchitecture.SEPARATED_ATTENTION_SKIP:
                pred_pose = dt_pose + pred_pose
            return torch.hstack((pred_trans, pred_shape, pred_pose))


    def _determine_input_output_size_pose(self) -> Tuple[int, int]:
        """
            Returns the number of pose parameters for the input and target model types.
            This is dependent on the rotation representation that the network should learn.

            Returns
            -------
                int:
                    Number of input pose parameters
                int:
                    Number of output pose parameters
        """
        return int(MODEL_STATS[self.model_from].joints * self.input_rotation_representation.get_number_components()), int(MODEL_STATS[self.model_to].joints * self.output_rotation_representation.get_number_components())
