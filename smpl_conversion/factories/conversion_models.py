# Copyright (c) 2025 Julien Fischer, Chair of Computer Graphics and Visualization, TUD Dresden University of Technology.
# All rights reserved.
# Licensed under the MIT License. See LICENSE in the project root for license information.

import torch
from dotmap import DotMap
from torch import nn
from typing import Union, Dict
from smpl_conversion.utils.enum_configurations import (
    ConversionNetworkArchitecture,
    BodyModelType,
    NetworkNormalizationMode,
    NetworkActivationFunction,
    PoseRepresentation
)
from smpl_conversion.models import (
    CombinedConversionNetwork,
    SeparatedConversionNetwork,
    SeparatedPerJointConversionNetwork
)

class ConversionModelFactory:
    """
        Creates conversion models
    """
    @staticmethod
    def create_conversion_model(config: Union[Dict, DotMap],
                                device: Union[torch.device, str]
                                ) -> nn.Module:
        """
            Creates and returns a body model from the provided config on the
            provided device
        """
        model_architecture = ConversionNetworkArchitecture.from_string(config['conversion']['model_type'])
        if model_architecture is None:
            raise ValueError(f"Requested model architecture is unknown: {config['conversion']['model_type']}")
        model_from, model_to = config['conversion']['mode'].split('2')
        input_body_type = BodyModelType.from_string(model_from)
        target_body_type = BodyModelType.from_string(model_to)
        if input_body_type is None or target_body_type is None:
            raise ValueError(f"Config contained unknown input or target body model type. Got {model_from} and {model_to}")
        n_betas = config['data']['n_shape_components']
        network_norm_layer = NetworkNormalizationMode.from_string(config['conversion']['network_norm_layer'])
        if network_norm_layer is None:
            raise ValueError(f"Config contained unknown network normalization layer: {config['conversion']['network_norm_layer']}")
        network_act_fct = NetworkActivationFunction.from_string(config['conversion'].get('network_activation', 'leaky_relu'))
        if network_act_fct is None:
            raise ValueError(f"Config contained unknown network activation function: {config['conversion']['network_activation']}")
        input_rotation_representation = PoseRepresentation.from_string(config['conversion'].get('input_rotation_representation', 'rot_vec'))
        if input_rotation_representation is None:
            raise ValueError(f"Config contained unknown input rotation representation: {config['conversion']['input_rotation_representation']}")
        target_rotation_representation = PoseRepresentation.from_string(config['conversion'].get('output_rotation_representation', 'rot_vec'))
        if target_rotation_representation is None:
            raise ValueError(f"Config contained unknown output rotation representation: {config['conversion']['output_rotation_representation']}")
        print(f"Creating new model of type {model_architecture}")
        if model_architecture.is_combined_architecture():
            model = CombinedConversionNetwork(
                input_body_type,
                target_body_type,
                n_betas,
                model_architecture,
                config['conversion']['model_max_dims'],
                network_norm_layer,
                network_act_fct,
                input_rotation_representation,
                target_rotation_representation
            )
        elif model_architecture.is_separated_architecture():
            model = SeparatedConversionNetwork(
                input_body_type,
                target_body_type,
                n_betas,
                model_architecture,
                config['conversion']['model_max_dims'],
                network_norm_layer,
                network_act_fct,
                input_rotation_representation,
                target_rotation_representation,
                config.get('transformer_settings', None)
            )
        elif model_architecture in [ConversionNetworkArchitecture.SEPARATED_PER_JOINT,
                                    ConversionNetworkArchitecture.SEPARATED_PER_JOINT_COMBINED]:
            model = SeparatedPerJointConversionNetwork(
                input_body_type,
                target_body_type,
                n_betas,
                target_rotation_representation,
                network_norm_layer,
                network_act_fct,
                config['conversion']['translation_network_hidden_size'],
                config['conversion'].get('translation_network_hidden_layers', 0),
                config['conversion']['shape_network_hidden_size'],
                config['conversion'].get('shape_network_hidden_layers', 0),
                config['conversion']['pose_network_hidden_size'],
                config['conversion'].get('pose_network_hidden_layers', 0),
                config['conversion']['use_residual_connections'],
                model_architecture==ConversionNetworkArchitecture.SEPARATED_PER_JOINT_COMBINED,
                config['conversion'].get('separated_combined_layer_size', -1),
                config['conversion'].get('separated_combined_layer_hidden_layers', -1),
                config['conversion'].get('separated_combined_layer_residual_connection', False)
            )
        else:
            raise ValueError(f"Found no constructor function for model architecture {model_architecture}")
        model = model.to(device)
        return model
