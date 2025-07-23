# Copyright (c) 2025 Julien Fischer, Chair of Computer Graphics and Visualization, TUD Dresden University of Technology.
# All rights reserved.
# Licensed under the MIT License. See LICENSE in the project root for license information.

import torch
import smplx
import numpy as np

from supr.pytorch.supr import SUPR
from typing import Union, Tuple, Dict, List
from smpl_conversion.utils.enum_configurations import BodyModelType, PoseRepresentation
from smpl_conversion.utils.parameter_processing import split_pose_parameters

def get_model_vertices_faces(body_model: torch.nn.Module,
                             pose_representation: PoseRepresentation,
                             parameters: Dict[str, Union[torch.Tensor, np.ndarray]],
                             return_as_tensor: bool = False,
                             model_device: Union[str, torch.device] = None
                             ) -> Tuple[
                                 Union[torch.Tensor, np.ndarray],
                                 Union[torch.Tensor, np.ndarray]
                             ]:
    """
        Evaluates the given body model and returns its vertices and faces.

        Params
        ------
            body_model (torch.nn):
                Body model that should be evaluated
            pose_representation (PoseRepresentation):
                Rotation representation of pose parameters.
            parameters (Dict):
                Parameters for which the body model should be evaluated.
                Is expected to contain keys 'trans', 'betas', 'poses' with
                torch.Tensor or np.ndarray entries that represent the parameters
            return_as_tensor (bool):
                Whether the resulting vertices and faces should be returned as
                tensors (True) or np.ndarrays (False). Defaults to False
            model_device (str or torch.device):
                If provided, transferes the parameters to the specified device
                before evaluation is performed. Defaults to None

        Returns
        -------
            torch.Tensor or np.ndarray:
                The vertices
            torch.Tensor or np.ndarray:
                The faces

        Raises
        ------
            NotImplementedError:
                If no evaluation procedure for the provided body_model
                is implemented.
    """
    if model_device is None:
        if isinstance(parameters['trans'], np.ndarray):
            model_device = "cpu"
        else:
            model_device = parameters['trans'].device
    model_type = BodyModelType.from_body_model_instance(body_model)
    if model_type is None:
        raise NotImplementedError(f"No evaluation procedure available for body model of type {type(body_model)}")

    model_params = {
        'betas': parameters['betas'].to(device=model_device, dtype=torch.float32) if isinstance(parameters['betas'], torch.Tensor) else torch.tensor(parameters['betas']).to(device=model_device, dtype=torch.float32)
    }
    if model_type in [BodyModelType.SMPL, BodyModelType.SMPLH, BodyModelType.SMPLX]:
        poses_split = split_pose_parameters(model_type,
                                            pose_representation,
                                            parameters['poses'])
        if isinstance(poses_split, dict):
            for key, value in poses_split.items():
                poses_split[key] = value.to(device=model_device, dtype=torch.float32)
        else:
            poses_split = poses_split.to(device=model_device, dtype=torch.float32)
        model_params.update(poses_split)
        model_params['transl'] = parameters['trans'].to(device=model_device, dtype=torch.float32) if isinstance(parameters['trans'], torch.Tensor) else torch.tensor(parameters['trans']).to(device=model_device, dtype=torch.float32)
    elif model_type in [BodyModelType.SUPR, BodyModelType.STAR]:
        model_params['pose'] = parameters['poses'].to(device=model_device, dtype=torch.float32) if isinstance(parameters['poses'], torch.Tensor) else torch.tensor(parameters['poses']).to(device=model_device, dtype=torch.float32)
        model_params['trans'] = parameters['trans'].to(device=model_device, dtype=torch.float32) if isinstance(parameters['trans'], torch.Tensor) else torch.tensor(parameters['trans']).to(device=model_device, dtype=torch.float32)
    else:
        raise ValueError(f"Body model type {model_type} not supported!")

    model_output = body_model(**model_params)
    vertices = model_output[:] if model_type in [BodyModelType.SUPR, BodyModelType.STAR] else model_output.vertices
    if not return_as_tensor and isinstance(vertices, torch.Tensor):
        vertices = vertices.detach().cpu().numpy()
    if return_as_tensor and isinstance(vertices, np.ndarray):
        vertices = torch.from_numpy(vertices)
    
    faces = body_model.faces
    if return_as_tensor and isinstance(faces, np.ndarray):
        faces = torch.from_numpy(faces)
    if not return_as_tensor and isinstance(faces, torch.Tensor):
        faces = faces.detach().cpu().numpy()
    return vertices, faces



def get_model_vertices_fast(body_model: torch.nn.Module,
                            pose_representation: PoseRepresentation,
                            parameters: Dict[str, torch.Tensor],
                            ) -> torch.Tensor:
    """
        Evaluates the given body model on the given parameters and returns the vertices.
        No checks are performed.

        Params
        ------
            body_model (torch.nn.Module):
                The body model which should be evaluated on the given parameters
            pose_representation (PoseRepresentation):
                Rotation representation of pose parameters
            parameters (Dict[str, torch.Tensor]):
                Dict-like that contains the parameters with keys 'trans', 'betas', 'poses'

        Returns
        -------
            torch.Tensor:
                The vertices of the resulting body model representation
    """
    model_type = BodyModelType.from_body_model_instance(body_model)
    params = {
        'betas': parameters['betas']
    }
    if model_type in [BodyModelType.SMPL, BodyModelType.SMPLH, BodyModelType.SMPLX]:
        poses_split = split_pose_parameters(model_type,
                                            pose_representation,
                                            parameters['poses'])
        params.update(poses_split)
        params['transl'] = parameters['trans']
    elif model_type in [BodyModelType.SUPR, BodyModelType.STAR]:
        params['pose'] = parameters['poses']
        params['trans'] = parameters['trans']
    model_output = body_model(**params)
    return model_output[:] if model_type in [BodyModelType.SUPR, BodyModelType.STAR] else model_output.vertices



def __check_obj_instance_of(obj,
                            instances: List) -> bool:
    """Checks whether obj is an instance of any of the provided list of instances"""
    is_instance = []
    for c in instances:
        is_instance.append(isinstance(obj, c))
    return any(is_instance)
