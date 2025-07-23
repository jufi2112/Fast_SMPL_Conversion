# Copyright (c) 2025 Julien Fischer, Chair of Computer Graphics and Visualization, TUD Dresden University of Technology.
# All rights reserved.
# Licensed under the MIT License. See LICENSE in the project root for license information.

import torch

from torch import nn
from typing import Union
from smpl_conversion.utils.enum_configurations import (
    NetworkActivationFunction, NetworkNormalizationMode,
    PoseRepresentation, BodyModelType
)
from smpl_conversion.utils.parameter_processing import direct_transfer_pose_parameters
from smpl_conversion.utils.model_infos import MODEL_STATS
from smpl_conversion.data.transforms import get_rotation_representation_conversion, PoseRepresentationConverter


class JointRotationConvertor(nn.Module):
    def __init__(self,
                 input_output_size: int,
                 hidden_size: int,
                 n_hidden_layers: int,
                 normalization_mode: NetworkNormalizationMode,
                 activation_function: NetworkActivationFunction,
                 use_residual_connection: bool
                 ):
        """
            Class that learns the rotation conversion of a single joint

            Params
            ------
                input_output_size (int):
                    Size of input and output layer
                hidden_size (int):
                    Size of the intermediate layer
                n_hidden_layers (int):
                    Number of hidden layers
                normalization_mode (NetworkNormalizationMode):
                    Normalization layer that should be used
                activation_function (NetworkActivationFunction):
                    Activation function that should be used
                use_residual_connection (bool):
                    Whether a residual connection should be employed
        """
        super().__init__()
        self.use_residual = use_residual_connection
        norm_layer = normalization_mode.get_pytorch_layer_class()
        act_fct = activation_function.get_pytorch_layer_class()
        input_layers = [
            nn.Linear(input_output_size, hidden_size),
            norm_layer(hidden_size) if normalization_mode == NetworkNormalizationMode.BATCH_NORM else act_fct(),
            act_fct() if normalization_mode == NetworkNormalizationMode.BATCH_NORM else norm_layer(hidden_size)
        ]
        output_layers = [nn.Linear(hidden_size, input_output_size)]
        layers = [
            *input_layers,
            *[hidden_layers for hidden_idx in range(n_hidden_layers) for hidden_layers in [
                nn.Linear(hidden_size, hidden_size),
                norm_layer(hidden_size) if normalization_mode == NetworkNormalizationMode.BATCH_NORM else act_fct(),
                act_fct() if normalization_mode == NetworkNormalizationMode.BATCH_NORM else norm_layer(hidden_size)
            ]],
            *output_layers
        ]
        self.network = nn.Sequential(*layers)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_residual:
            return x + self.network(x)
        else:
            return self.network(x)



class ShapeConvertor(nn.Module):
    def __init__(self,
                 n_shape_components: int,
                 hidden_size: int,
                 n_hidden_layers: int,
                 normalization_mode: NetworkNormalizationMode,
                 activation_function: NetworkActivationFunction,
                 use_residual_connection: bool):
        """
            Class that learns shape parameter conversion
        
            Params
            ------
                n_shape_components (int):
                    Number of shape components that should be converted
                n_hidden_layers (int):
                    Size of the hidden layer
                hidden_layers (int):
                    Number of hidden layers
                normalization_mode (NetworkNormalizationMode):
                    Normalization layer that should be used
                activation_function (NetworkActivationFunction):
                    Activation function that should be used
                use_residual_connection (bool):
                    Whether a residual connection should be used
        """
        super().__init__()
        self.use_residual = use_residual_connection
        norm_layer = normalization_mode.get_pytorch_layer_class()
        act_fct = activation_function.get_pytorch_layer_class()
        input_layers = [
            nn.Linear(n_shape_components, hidden_size),
            norm_layer(hidden_size) if normalization_mode == NetworkNormalizationMode.BATCH_NORM else act_fct(),
            act_fct() if normalization_mode == NetworkNormalizationMode.BATCH_NORM else norm_layer(hidden_size),
        ]
        output_layers = [nn.Linear(hidden_size, n_shape_components)]
        layers = [
            *input_layers,
            *[hidden_layers for hidden_idx in range(n_hidden_layers) for hidden_layers in [
                nn.Linear(hidden_size, hidden_size),
                norm_layer(hidden_size) if normalization_mode == NetworkNormalizationMode.BATCH_NORM else act_fct(),
                act_fct() if normalization_mode == NetworkNormalizationMode.BATCH_NORM else norm_layer(hidden_size)
            ]],
            *output_layers
        ]
        self.network = nn.Sequential(*layers)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_residual:
            return x + self.network(x)
        else:
            return self.network(x)



class TranslationConvertor(nn.Module):
    def __init__(self,
                 hidden_size: int,
                 n_hidden_layers: int,
                 normalization_mode: NetworkNormalizationMode,
                 activation_function: NetworkActivationFunction,
                 use_residual_connection: bool):
        """
            Class that learns translation parameter conversion

            Params
            ------
                hidden_size (int):
                    Size of the hidden layer
                n_hidden_layers (int):
                    Number of hidden layers
                normalization_mode (NetworkNormalizationMode):
                    Normalization layer that should be used
                activation_function (NetworkActivationFunction):
                    Activation function that should be used
                use_residual_connection (bool):
                    Whether a residual connection should be used
        """
        super().__init__()
        self.use_residual = use_residual_connection
        norm_layer = normalization_mode.get_pytorch_layer_class()
        act_fct = activation_function.get_pytorch_layer_class()
        input_layers = [
            nn.Linear(3, hidden_size),
            norm_layer(hidden_size) if normalization_mode == NetworkNormalizationMode.BATCH_NORM else act_fct(),
            act_fct() if normalization_mode == NetworkNormalizationMode.BATCH_NORM else norm_layer(hidden_size)
        ]
        output_layers = [nn.Linear(hidden_size, 3)]
        layers = [
            *input_layers,
            *[hidden_layers for hidden_idx in range(n_hidden_layers) for hidden_layers in [
                nn.Linear(hidden_size, hidden_size),
                norm_layer(hidden_size) if normalization_mode == NetworkNormalizationMode.BATCH_NORM else act_fct(),
                act_fct() if normalization_mode == NetworkNormalizationMode.BATCH_NORM else norm_layer(hidden_size)
            ]],
            *output_layers
        ]
        self.network = nn.Sequential(*layers)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_residual:
            return x + self.network(x)
        else:
            return self.network(x)



class PerJointPoseConvertor(nn.Module):
    def __init__(self,
                 model_from: BodyModelType,
                 model_to: BodyModelType,
                 pose_representation: PoseRepresentation,
                 normalization_mode: NetworkNormalizationMode,
                 activation_function: NetworkActivationFunction,
                 network_hidden_size: int,
                 network_hidden_layers: int,
                 use_residual_connections: bool
                 ):
        """
            Conversion network that separately converts each joint rotation

            Params
            ------
                model_from (BodyModelType):
                    Body model type from which should be converted
                model_to (BodyModelType):
                    Body model type to which should be converted
                pose_representation (PoseRepresentation):
                    Pose representation which should be used during conversion.
                    Input and output parameters are always rotation vectors
                normalization_mode (NetworkNormalizationMode):
                    Normalization mode that should be applied
                activation_function (NetworkActivationFunction):
                    Activation function that should be applied
                network_hidden_size (int):
                    Hidden size of each joint rotation network
                network_hidden_layers (int):
                    Number of hidden layers for each joint rotation network
                use_residual_connections (bool):
                    Whether residual connections should be used
        """
        super().__init__()
        self.model_from = model_from
        self.model_to = model_to
        self.pose_representation = PoseRepresentation.ROTATION_VECTOR # pose_representation
        self.use_residual_connections = use_residual_connections
        
        self.n_target_joints = MODEL_STATS[self.model_to]['joints']
        input_output_size = self.pose_representation.get_number_components()

        # A joint rotation convertor network for each joint in the target
        # body model hierarchy
        self.networks = nn.ModuleList([
            JointRotationConvertor(input_output_size,
                                   network_hidden_size,
                                   network_hidden_layers,
                                   normalization_mode,
                                   activation_function,
                                   self.use_residual_connections)
            for _ in range(self.n_target_joints)
        ])

        # transforms rotation representation from intermediate (used inside of
        # networks) back to rotation vector

        # TODO: If rot repr is not rotation vector:
        # 1. Make sure output is a valid rotation representation
        # 2. Convert output to rotation matrix
        #       This operation needs to be differentiable
        # self.rot_repr_convertor: PoseRepresentationConverter = get_rotation_representation_conversion(pose_representation,
        #                                                                                               PoseRepresentation.ROTATION_VECTOR)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
            Assumes x to be of shape (N, n_joints_input_model * 3).
            The resulting tensor will be of shape (N, n_joints_target_model * 3)
        """
        # Transfer input pose parameters to target pose parameters in "naive" way
        # (this adds / removes joints that are present / not present in target body model type)

        # Conversion of input data from one pose representation to another does not
        # have to be differentiable
        pose_direct_transfer = direct_transfer_pose_parameters(self.model_from,
                                                               self.model_to,
                                                               PoseRepresentation.ROTATION_VECTOR,
                                                               PoseRepresentation.ROTATION_VECTOR,
                                                               #self.pose_representation,
                                                               x)
        assert pose_direct_transfer.shape[1] / self.pose_representation.get_number_components() == self.n_target_joints, \
            f"Directly transferred pose shape does not match expected shape {pose_direct_transfer.shape[1]} vs {self.pose_representation.get_number_components() * self.n_target_joints}"
        # Put each joint's rotation through the corresponding network and combine results
        conversion_results = []
        pose_components = 3 # self.pose_representation.get_number_components()
        for joint_idx in range(self.n_target_joints):
            conversion_results.append(
                self.networks[joint_idx](pose_direct_transfer[:, joint_idx * pose_components : (joint_idx + 1) * pose_components])
            )
        conversion_results = torch.cat(conversion_results, dim=1)
        # Convert back to rotation vector representation
        #res = self.rot_repr_convertor(conversion_results)
        return conversion_results



class SeparatedPerJointCombinedLayer(nn.Module):
    def __init__(self,
                 output_model_type: BodyModelType,
                 n_shape_components: int,
                 pose_representation: PoseRepresentation,
                 normalization_mode: NetworkNormalizationMode,
                 activation_function: NetworkActivationFunction,
                 n_hidden_layers: int,
                 hidden_layers_size: int):
        """
            Applies a combined conversion layer. Does not use any residual connections
            by itself.

            Params
            ------
                output_model_type (BodyModelType):
                    Target model type
                n_shape_components (int):
                    Number of shape components
                pose_representation (PoseRepresentation):
                    Rotation representation of pose parameters
                normalization_mode (NetworkNormalizationMode):
                    Normalization mode
                activation_function (NetworkActivationFunction):
                    Activation function
                n_hidden_layers (int):
                    Number of hidden layers
                hidden_layers_size (int):
                    Size of each hidden layer. -1 corresponds to same as input
                    size
        """
        super().__init__()
        norm_layer = normalization_mode.get_pytorch_layer_class()
        act_fct = activation_function.get_pytorch_layer_class()
        n_pose_params = MODEL_STATS[output_model_type]['joints'] * pose_representation.get_number_components()
        input_output_size = 3 + n_shape_components + n_pose_params
        if hidden_layers_size == -1:
            hidden_layers_size = input_output_size
        input_layers = [
            nn.Linear(input_output_size, hidden_layers_size),
            norm_layer(hidden_layers_size) if normalization_mode == NetworkNormalizationMode.BATCH_NORM else act_fct(),
            act_fct() if normalization_mode == NetworkNormalizationMode.BATCH_NORM else norm_layer(hidden_layers_size)
        ]
        output_layers = [nn.Linear(hidden_layers_size, input_output_size)]
        layers = [
            *input_layers,
            *[hidden_layers for hidden_idx in range(n_hidden_layers) for hidden_layers in [
                nn.Linear(hidden_layers_size, hidden_layers_size),
                norm_layer(hidden_layers_size) if normalization_mode == NetworkNormalizationMode.BATCH_NORM else act_fct(),
                act_fct() if normalization_mode == NetworkNormalizationMode.BATCH_NORM else norm_layer(hidden_layers_size)
            ]],
            *output_layers
        ]
        self.network = nn.Sequential(*layers)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)



class SeparatedPerJointConversionNetwork(nn.Module):
    def __init__(self,
                 model_from: BodyModelType,
                 model_to: BodyModelType,
                 n_shape_components: int,
                 pose_representation: PoseRepresentation,
                 normalization_mode: NetworkNormalizationMode,
                 activation_function: NetworkActivationFunction,
                 translation_network_hidden_size: int,
                 translation_network_hidden_layers: int,
                 shape_network_hidden_size: int,
                 shape_network_hidden_layers: int,
                 pose_network_hidden_size: int,
                 pose_network_hidden_layers: int,
                 use_residual_connections: bool,
                 append_combined_layer: bool,
                 combined_layer_hidden_size: int = 0,
                 combined_layer_hidden_layers: int = 0,
                 combined_layer_use_residual_connection: bool = True
                 ):
        """
            Converts pose, translation, and shape parameters. Pose parameters of
            each joint are converted separately.

            Params
            ------
                model_from (str):
                    Model type of input parameters
                model_to (str):
                    Target model type
                n_shape_components (int):
                    Number of shape parameters (betas)
                pose_representation (PoseRepresentation):
                    Pose (rotation) representation that should be used inside
                    of the networks
                normalization_mode (NetworkNormalizationMode):
                    Type of normalization that should be used in the networks
                activation_function (NetworkActivationFunction):
                    Activation function that should be used in the networks
                translation_network_hidden_size (int):
                    Hidden size of the translation network
                translation_network_hidden_layers (int):
                    Number of hidden layers of the translation network
                shape_network_hidden_size (int):
                    Hidden size of the shape network
                shape_network_hidden_layers (int):
                    Number of hidden layers of the translation network
                pose_network_hidden_size (int):
                    Hidden size of the pose networks
                pose_network_hidden_layers (int):
                    Number of hidden layers of the pose networks
                use_residual_connections (bool):
                    Whether the different networks should employ residual
                    connections
                append_combined_layer (bool):
                    If True, adds a combined conversion network at the end of
                    the network
                combined_layer_hidden_size (int):
                    Size of each hidden layer in the combined network.
                    Optional, defaults to 0
                combined_layer_hidden_layers (int):
                    Number of hidden layers in the combined network.
                    Optional, defaults to 0
                combined_layer_use_residual_connection (bool):
                    Whether a potential combined layer should use a residual
                    connection. Optional, defaults to True
        """
        super().__init__()
        self.model_from = model_from
        self.model_to = model_to
        self.n_shape_components = n_shape_components
        self.pose_representation = PoseRepresentation.ROTATION_VECTOR # pose_representation
        self.normalization_mode = normalization_mode
        self.activation_function = activation_function
        self.translation_network_hidden_size = translation_network_hidden_size
        self.translation_network_hidden_layers = translation_network_hidden_layers
        self.shape_network_hidden_size = shape_network_hidden_size
        self.shape_network_hidden_layers = shape_network_hidden_layers
        self.pose_network_hidden_size = pose_network_hidden_size
        self.pose_network_hidden_layers = pose_network_hidden_layers
        self.use_residual_connections = use_residual_connections
        self.append_combined_layer = append_combined_layer
        self.combined_network_hidden_layers = combined_layer_hidden_layers
        self.combined_network_hidden_size = combined_layer_hidden_size
        self.combined_network_use_residual_connection = combined_layer_use_residual_connection

        self.translation_network = TranslationConvertor(self.translation_network_hidden_size,
                                                        self.translation_network_hidden_layers,
                                                        self.normalization_mode,
                                                        self.activation_function,
                                                        self.use_residual_connections
                                                        )
        self.shape_network = ShapeConvertor(self.n_shape_components,
                                            self.shape_network_hidden_size,
                                            self.shape_network_hidden_layers,
                                            self.normalization_mode,
                                            self.activation_function,
                                            self.use_residual_connections
                                            )
        self.pose_network = PerJointPoseConvertor(self.model_from,
                                                  self.model_to,
                                                  self.pose_representation,
                                                  self.normalization_mode,
                                                  self.activation_function,
                                                  self.pose_network_hidden_size,
                                                  self.pose_network_hidden_layers,
                                                  self.use_residual_connections
                                                  )
        if self.append_combined_layer:
            self.combined_network = SeparatedPerJointCombinedLayer(self.model_to,
                                                                   self.n_shape_components,
                                                                   self.pose_representation,
                                                                   self.normalization_mode,
                                                                   self.activation_function,
                                                                   self.combined_network_hidden_layers,
                                                                   self.combined_network_hidden_size
                                                                   )
        self.print_information()


    def print_information(self):
        print("Separated-Per-Joint-Conversion-Network Information")
        print("==================================================")
        print(f"Input model type: {str(self.model_from)}")
        print(f"Target model type: {str(self.model_to)}")
        print(f"Number of shape components: {self.n_shape_components}")
        print(f"Internal pose representation: {str(self.pose_representation)}")
        print(f"Network normalization mode: {str(self.normalization_mode)}")
        print(f"Network activation function: {str(self.activation_function)}")
        print(f"Networks use residual connections: {self.use_residual_connections}")
        print(f"Translation Network:")
        print(f"   Hidden Layers: {self.translation_network_hidden_layers}")
        print(f"   Hidden Layer Size: {self.translation_network_hidden_size}")
        print(f"Shape Network:")
        print(f"   Hidden Layers: {self.shape_network_hidden_layers}")
        print(f"   Hidden Layer Size: {self.shape_network_hidden_size}")
        print(f"Pose Network:")
        print(f"   Hidden Layers: {self.pose_network_hidden_layers}")
        print(f"   Hidden Layer Size: {self.pose_network_hidden_size}")
        print(f"Employ trailing combined conversion network: {self.append_combined_layer}")
        if self.append_combined_layer:
            print(f"Combined Network:")
            print(f"   Hidden Layers: {self.combined_network_hidden_layers}")
            print(f"   Hidden Layer Size: {self.combined_network_hidden_size}")
            print(f"   Use Residual Connection: {self.combined_network_use_residual_connection}")
        print("==================================================")


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
            Assumes x to be of shape (B, 3 + n_shape_components + M) where
            M = n_joints * size_pose_representation (e.g. 3 for rotation vector,
            9 for default rotation matrix).
            The returned tensor is of shape (B, 3 + n_shape_components + N)
            where N = n_joints * size_output_pose_representation
        """
        trans = x[:, :3]
        betas = x[:, 3: 3+self.n_shape_components]
        poses = x[:, 3+self.n_shape_components:]

        pred_trans = self.translation_network(trans)
        pred_betas = self.shape_network(betas)
        pred_poses = self.pose_network(poses)

        pred = torch.cat((pred_trans, pred_betas, pred_poses), dim=1)
        if self.append_combined_layer:
            if self.combined_network_use_residual_connection:
                pred = pred + self.combined_network(pred)
            else:
                pred = self.combined_network(pred)
        return pred
