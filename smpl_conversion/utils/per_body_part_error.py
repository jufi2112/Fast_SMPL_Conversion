# Copyright (c) 2025 Julien Fischer, Chair of Computer Graphics and Visualization, TUD Dresden University of Technology.
# All rights reserved.
# Licensed under the MIT License. See LICENSE in the project root for license information.

import os
import json
import torch
import numpy as np

from os import path as osp
from typing import Dict, List, Union
from smpl_conversion.utils.enum_configurations import BodyPart, BodyModelType, MetricReturnType

class PerBodyPartError:
    def __init__(self,
                 model_type: BodyModelType,
                 body_model_location: str):
        """
            Calculates a per-body-part error from a given whole-body error
        """
        self.model_type = model_type
        self.body_part_segmentation = self._load_body_part_segmentations(osp.join(body_model_location, 'segmentations'))
        self.supported_body_parts = list(self.body_part_segmentation.keys())


    def _load_body_part_segmentations(self, dirpath: str) -> Dict[str, List[int]]:
        if self.model_type in [BodyModelType.SMPL, BodyModelType.SMPLH,
                               BodyModelType.STAR, BodyModelType.SKEL]:
            # Load smpl segmentation
            fname = "smpl_vert_segmentation.json"
        elif self.model_type in [BodyModelType.SMPLX, BodyModelType.SUPR]:
            # Load smplx segmentation
            fname = "smplx_vert_segmentation.json"
        fpath = osp.join(dirpath, fname)
        if not osp.exists(fpath):
            raise ValueError(f"Could not find file {fname} in directory {dirpath}")
        raw_segmentations = None
        segmentations = {}
        with open(fpath) as file:
            raw_segmentations = json.load(file)
        if not isinstance(raw_segmentations, dict):
            raise ValueError(f"The loaded segmentation is not of type dict but {type(raw_segmentations)}")
        for key, value in raw_segmentations.items():
            key_enum = BodyPart.from_string(key)
            if key_enum is None:
                raise ValueError(f"Unrecognized body part: {key}")
            segmentations[key_enum] = value
        return segmentations


    def _mean(self, arr: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        if isinstance(arr, np.ndarray):
            return np.mean(arr, axis=-1)
        else:
            return torch.mean(arr, dim=-1)


    def _sum(self, arr: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        if isinstance(arr, np.ndarray):
            return np.sum(arr, axis=-1)
        else:
            return torch.sum(arr, dim=-1)


    def get_supported_body_parts(self) -> List[BodyPart]:
        return self.supported_body_parts


    def get_per_body_part_errors(self,
                                 per_vertex_errors: Union[np.ndarray, torch.Tensor],
                                 body_parts: Union[BodyPart, List[BodyPart]],
                                 return_type: Union[str, MetricReturnType] = 'Mean'
                                 ) -> Dict[BodyPart, Union[np.ndarray, torch.Tensor]]:
        """
            For a given whole body error, calculates the per-body part errors
            for the requested body parts.

            Params
            ------
                per_vertex_errors (np.ndarray or torch.Tensor):
                    An array / tensor of shape (B, V) that contains the
                    per-vertex errors of the whole body.
                body_parts (BodyPart or list of BodyPart):
                    The body parts for which the errors should be calculated
                return_type (str or MetricReturnType):
                    How the per-body part errors should be returned. 'All'
                    (returns the error for each vertex that belongs to the body
                    part), 'Mean', or 'Sum'. Defaults to 'Mean'

            Returns
            -------
                dict of BodyPart and np.ndarray or torch.Tensor:
                    Error for each body part
        """
        if isinstance(body_parts, BodyPart):
            body_parts = [body_parts]
        if not isinstance(return_type, MetricReturnType):
            return_type = MetricReturnType.from_string(return_type)
            if return_type is None:
                raise ValueError("Unrecognized option for argument 'return_type'")
        per_part_errors = {}
        for body_part in body_parts:
            if body_part not in self.supported_body_parts:
                raise ValueError(f"The requested body part '{str(body_part)}' is not supported by model type {self.model_type}")
            # Get losses for each vertex that is part of the current body part
            body_part_errors = per_vertex_errors[:, self.body_part_segmentation[body_part]]
            if return_type == MetricReturnType.ALL:
                per_part_errors[body_part] = body_part_errors
            elif return_type == MetricReturnType.MEAN:
                per_part_errors[body_part] = self._mean(body_part_errors)
            elif return_type == MetricReturnType.SUM:
                per_part_errors[body_part] = self._sum(body_part_errors)
            else:
                raise ValueError(f"Unknown option for MetricReturnType: {return_type.__repr__()}")
        return per_part_errors
