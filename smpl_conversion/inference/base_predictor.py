# Copyright (c) 2025 Julien Fischer, Chair of Computer Graphics and Visualization, TUD Dresden University of Technology.
# All rights reserved.
# Licensed under the MIT License. See LICENSE in the project root for license information.

import torch

from dotmap import DotMap
from os import path as osp
from abc import ABC, abstractmethod
from typing import Union, Dict, Tuple
from smpl_conversion.data.transforms import get_data_transform_class
from smpl_conversion.utils.enum_configurations import PoseRepresentation, BodyModelType

class BasePredictor(ABC):
    def __init__(self,
                 checkpoint_location: str,
                 device: str = 'auto',
                 verbosity: int = 0
                 ):
        """
            Base class for prediction classes

            Params
            ------
                checkpoint_location (str):
                    Path to the checkpoint that should be loaded
                device (str):
                    Device where prediction and possibly metric calculations
                    should be performed on. Defaults to 'auto', i.e. CUDA if
                    torch.cuda.is_available() is true, otherwise CPU
                verbosity (int):
                    Verbosity level, defaults to 0.
        """
        if device == 'auto':
            if torch.cuda.is_available():
                device = 'cuda'
            else:
                device = 'cpu'
        self.device = device
        self.verbosity = verbosity
        (
            self.model_from, self.model_to, self.gender, self.model,
            self.normalizer, self.config, self.input_rotation_representation,
            self.output_rotation_representation, self.supr_constrained
        ) = self._load_checkpoint(checkpoint_location)
        if self.verbosity > 0:
            print("Loaded checkpoint with following information:")
            print(f"   Conversion from: {self.model_from}")
            print(f"   Conversion to: {self.model_to}")
            print(f"   Gender: {self.gender}")
            print(f"   Data normalization: {'No' if self.normalizer is None else 'Yes'}")
            print(f"   Model architecture: {self.model}")
            print(f"   Network input rotation representation: {str(self.input_rotation_representation)}")
            print(f"   Network output rotation representation: {str(self.output_rotation_representation)}")
            if self.model_from == BodyModelType.SUPR or self.model_to == BodyModelType.SUPR:
                print(f"   Will use {'constrained' if self.supr_constrained else 'unconstrained'} SUPR model")


    def _load_checkpoint(self,
                         checkpoint_path: str
                         ) -> Tuple[BodyModelType, BodyModelType, str, torch.nn.Module,
                                    torch.nn.Module, Dict, PoseRepresentation,
                                    PoseRepresentation, bool]:
        """
            Loads the given checkpoint.

            Params
            ------
                checkpoint_path (str):
                    Path to the checkpoint that should be loaded.

            Returns
            -------
                BodyModelType:
                    Model from which should be converted
                BodyModelType:
                    Model to which should be converted
                str:
                    Gender
                torch.nn.Module:
                    Conversion network
                torch.nn.Module:
                    Normalizer
                Dict:
                    Configuration dictionary
                PoseRepresentation:
                    Rotation representation that the network uses
                PoseRepresentation:
                    Rotation representation that the network outputs
                torch.nn.Module:
                    Module that transforms dataset rotation representation to
                    the network's required rotation representation
                torch.nn.Module:
                    Module that transforms the network's output to the dataset
                    rotation representation (rotation vector)
                bool:
                    Whether a potentially used SUPR model should be constrained
        """
        if not osp.isfile(checkpoint_path):
            raise ValueError(f"Given path {checkpoint_path} is not a file!")
        ckpt = torch.load(checkpoint_path)
        config = ckpt['train_progress']['config']
        model_from, model_to = config.conversion.mode.split('2')
        supr_constrained = config.conversion.supr_is_constrained
        model_from = BodyModelType.from_string(model_from)
        model_to = BodyModelType.from_string(model_to)
        gender = config.general.gender

        # Parse pose rotation representation
        input_rotation_representation = PoseRepresentation.from_string(config['conversion'].get('input_rotation_representation', None))
        if input_rotation_representation is None:
            print("No or invalid entry for config field 'conversion.input_rotation_representation', defaulting to rotation vector representation!")
            input_rotation_representation = PoseRepresentation.ROTATION_VECTOR
            config['conversion']['input_rotation_representation'] = 'rotation_vector'
        output_rotation_representation = PoseRepresentation.from_string(config['conversion'].get('output_rotation_representation', None))
        if output_rotation_representation is None:
            print("No or invalid entry for config field 'conversion.output_rotation_representation', defaulting to rotation vector representation!")
            output_rotation_representation = PoseRepresentation.ROTATION_VECTOR
            config['conversion']['output_rotation_representation'] = 'rotation_vector'
        model = self._create_model(config)
        model.load_state_dict(ckpt['model_state_dict'])
        normalizers = ckpt.get('normalizers', None)
        if normalizers is None:
            normalizer = None
        else:
            raise NotImplementedError("Normalization is currently not supported")
            normalizer = {}
            for param in config.conversion.parameter:
                key = f"{model_from}_{param}_{gender}"
                if key in normalizers:
                    normalizer[param] = normalizers[key]
            if len(normalizer) == 0:
                raise ValueError(
                    f"Could not find suitable normalizers in {normalizers.keys()}, "
                    f"expected normalizers for parameters {config.conversion.parameter}"
                )
        return (
            model_from, model_to, gender, model, normalizer, config,
            input_rotation_representation, output_rotation_representation,
            supr_constrained
        )


    def get_input_rotation_rerpesentation(self) -> PoseRepresentation:
        return self.input_rotation_representation


    def get_output_rotation_representation(self) -> PoseRepresentation:
        return self.output_rotation_representation


    def get_input_model_type(self) -> BodyModelType:
        return self.model_from


    def get_output_model_type(self) -> BodyModelType:
        return self.model_to


    def get_gender(self) -> str:
        return self.gender


    def get_number_shape_components(self) -> int:
        return self.config['data'].get('n_shape_components', None)


    @abstractmethod
    def _create_model(self,
                      config: DotMap):
        pass


    @abstractmethod
    def predict(self, X):
        pass
