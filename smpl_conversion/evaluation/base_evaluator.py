# Copyright (c) 2025 Julien Fischer, Chair of Computer Graphics and Visualization, TUD Dresden University of Technology.
# All rights reserved.
# Licensed under the MIT License. See LICENSE in the project root for license information.

import torch

from dotmap import DotMap
from os import path as osp
from torch import nn as nn
from abc import ABC, abstractmethod
from typing import Tuple, Dict, List
from torch.utils.data import DataLoader
from smpl_conversion.data.transforms import get_data_transform_class
from smpl_conversion.utils.enum_configurations import PoseRepresentation, BodyModelType

class BaseEvaluator(ABC):
    def __init__(self,
                 checkpoint_location: str,
                 dataset_location: str,
                 body_model_location: str,
                 transfer_file_location: str,
                 device_dataset: str,
                 device_inference: str = 'auto',
                 verbosity: int = 0):
        self.checkpoint_fpath = checkpoint_location
        self.dataset_fpath = dataset_location
        self.device_dataset = device_dataset
        self.body_model_location = body_model_location
        self.transfer_file_location = transfer_file_location
        if isinstance(device_inference, str) and device_inference == 'auto':
            if torch.cuda.is_available():
                device_inference = "cuda"
            else:
                device_inference = "cpu"
        self.device_inference = device_inference
        self.verbosity = verbosity
        (
            self.model_from, self.model_to, self.gender,
            self.model, self.normalizer, self.config,
            self.input_rotation_representation,
            self.output_rotation_representation, self.input_rotation_transform,
            self.output_rotation_transform, self.supr_constrained
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
                print(f"   Using {'constrained' if self.supr_constrained else 'unconstrained'} SUPR model")
        if self.verbosity > 0:
            print("Loading dataset... ", end='')
        self.dataset = self._load_dataset(dataset_location)
        if self.verbosity > 0:
            print("done")
        self.data_loader = None


    def _load_checkpoint(self,
                         checkpoint_path: str
                         ) -> Tuple[BodyModelType, BodyModelType, str, nn.Module, None, Dict,
                                    PoseRepresentation, PoseRepresentation,
                                    nn.Module, nn.Module]:
        """
            Loads the checkpoint from the given file.

            Params
            ------
                checkpoint_path (str):
                    Path to the checkpoint file that should be loaded.

            Returns
            -------
                BodyModelType:
                    Model from which to convert
                BodyModelType:
                    Model to which to convert
                str:
                    Gender for which the conversion was trained
                nn.Module:
                    The trained model with weights already loaded.
                None:
                    The normalizer object, currently not supported
                Dict:
                    The configuration
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
                    Whether a potential SUPR model should be constrained


            Raises
            ------
                ValueError:
                    If the given path does not point to a valid file
        """
        if not osp.isfile(checkpoint_path):
            raise ValueError(f"Given path {checkpoint_path} is not a file!")
        ckpt = torch.load(checkpoint_path)
        config = ckpt['train_progress']['config']
        model_from, model_to = config.conversion.mode.split('2')
        model_from = BodyModelType.from_string(model_from)
        model_to = BodyModelType.from_string(model_to)
        gender = config.general.gender
        supr_constrained = config['conversion']['supr_is_constrained']

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
        input_rotation_transform, output_rotation_transform = get_data_transform_class(input_rotation_representation,
                                                                                       output_rotation_representation,
                                                                                       config['data']['n_shape_components']
                                                                                       )

        model = self._create_model(config,
                                   input_rotation_representation,
                                   output_rotation_representation)
        model.load_state_dict(ckpt['model_state_dict'])
        normalizers = ckpt.get('normalizers', None)
        if normalizers is None:
            normalizer = None
        else:
            raise ValueError("Normalization is currently not supported")
        return (
            model_from, model_to, gender, model, normalizer, config,
            input_rotation_representation, output_rotation_representation,
            input_rotation_transform, output_rotation_transform,
            supr_constrained
        )


    def _create_data_loader(self, batch_size: int):
        """
            Creates a torch dataloader for the dataset at self.dataset

        Params
        ------
            batch_size (int):
                Batch size to use.

        Raises
        ------
            ValueError:
                If self.dataset is None
        """
        if self.dataset is None:
            raise ValueError("Cannot create a data loader for None dataset")
        data_loader = DataLoader(dataset=self.dataset,
                                 batch_size=batch_size,
                                 shuffle=False
                                 )
        return data_loader


    @abstractmethod
    def _load_dataset(self, dataset_path: str):
        pass


    @abstractmethod
    def _create_model(self,
                      config: DotMap,
                      input_rotation_representation: PoseRepresentation,
                      output_rotation_representation: PoseRepresentation):
        pass


    @abstractmethod
    def evaluate(self, batch_size: int,
                 metric_names: List[str],
                 show_loss_distribution: bool):
        pass