# Copyright (c) 2025 Julien Fischer, Chair of Computer Graphics and Visualization, TUD Dresden University of Technology.
# All rights reserved.
# Licensed under the MIT License. See LICENSE in the project root for license information.

import torch
import numpy as np

from dotmap import DotMap
from typing import Union, Dict
from smpl_conversion.utils.model_mappings import direct_transfer_indices
from smpl_conversion.utils.model_infos import MODEL_STATS, MODELS
from smpl_conversion.utils.enum_configurations import BodyModelType, PoseRepresentation
from smpl_conversion.data.transforms import (
    get_rotation_representation_conversion,
    PoseRepresentationConverter
)

def split_pose_parameters(model_type: BodyModelType,
                          pose_representation: PoseRepresentation,
                          parameters: Union[torch.Tensor, np.ndarray]
                          ) -> DotMap[str, torch.Tensor]:
    """
        Splits pose parameters into separate tensors for global orientation and
        body pose, as well as hand and face. If model_type is SUPR or STAR, a single
        tensor is returned.

        Params
        ------
        model_type (BodyModelType):
            Body model type to which the parameters belong.
        pose_representation (PoseRepresentation):
            Rotation representation of pose parameters.
        parameters (torch.Tensor or np.ndarray):
            Full pose parameters that should be splitted

        Returns
        -------
            Dotmap[str, torch.Tensor]:
                A dotmap that contains the poses for the different body parts. The
                names correspond to the ones used in smpl/smplh/smplx/supr/star,
                such that they can be used directly.

        Note
        ----
            The returned splits have shape (B, J * x) where x is the number of
            rotation components of the given pose_representation (e.g. 3 for
            rotation vectors, 4 for quaternions, 6 for 6D continuous and 9
            for rotation matrix) and J is the number of joints
    """
    if isinstance(parameters, np.ndarray):
        parameters = torch.tensor(parameters, dtype=torch.float32)
    if len(parameters.shape) == 1:
        parameters = parameters.unsqueeze(0)
    bs = len(parameters)
    n_pose_components = pose_representation.get_number_components()
    poses = DotMap(_dynamic=False)
    poses.global_orient = parameters[:, :n_pose_components]
    if model_type == BodyModelType.SMPL:
        poses.body_pose = parameters[:, n_pose_components:]
    elif model_type == BodyModelType.SMPLH:
        poses.body_pose = parameters[:, n_pose_components:(22*n_pose_components)]
        poses.left_hand_pose = parameters[:, (22*n_pose_components):(37*n_pose_components)]
        poses.right_hand_pose = parameters[:, (37*n_pose_components):]
    elif model_type == BodyModelType.SMPLX:
        poses.body_pose = parameters[:, n_pose_components:(22*n_pose_components)]
        poses.jaw_pose = parameters[:, (22*n_pose_components):(23*n_pose_components)]
        poses.leye_pose = parameters[:, (23*n_pose_components):(24*n_pose_components)]
        poses.reye_pose = parameters[:, (24*n_pose_components):(25*n_pose_components)]
        poses.left_hand_pose = parameters[:, (25*n_pose_components):(40*n_pose_components)]
        poses.right_hand_pose = parameters[:, (40*n_pose_components):]
    elif model_type in [BodyModelType.SUPR, BodyModelType.STAR]:
        return DotMap({'pose': parameters}, _dynamic=False)
    else:
        raise NotImplementedError(f"Unsupported body model type: {model_type}")
    # If SMPL / SMPL+H / SMPL-X and rotation representation not rotation vector or quaternion, we have to reshape the pose parameters
    # if model_type not in [BodyModelType.SUPR, BodyModelType.STAR] and pose_representation in [PoseRepresentation.ROTATION_MATRIX_6D, PoseRepresentation.ROTATION_MATRIX_9D]:
    #     n_cols = 3 if pose_representation == PoseRepresentation.ROTATION_MATRIX_9D else 2
    #     for pose_key in poses:
    #         if pose_key in ['global_orient', 'jaw_pose']:
    #             # No joint dimension
    #             poses[pose_key] = poses[pose_key].reshape(bs, 3, n_cols)
    #         else:
    #             poses[pose_key] = poses[pose_key].reshape(bs, -1, 3, n_cols)
    return poses


def combine_pose_parameters(poses_dict: Dict[str, torch.Tensor]
                            ) -> torch.Tensor:
    """
        Combines suitable pose parameters from the provided dictionary
        into a joint tensor.

        Params
        ------
            poses_dict (Dict[str, torch.Tensor]):
                Potential separated pose tensors that should be joined.

        Returns
        -------
            torch.Tensor:
                The concatenated pose tensor
    """
    # Process keys to match expected shape
    for key in ['global_orient', 'body_pose', 'jaw_pose', 'leye_pose',
                'reye_pose', 'left_hand_pose', 'right_hand_pose']:
        if key in poses_dict.keys():
            bs = len(poses_dict[key])
            if poses_dict[key].ndim == 1:
                poses_dict[key] = torch.unsqueeze(poses_dict[key], dim=0)
            elif poses_dict[key].ndim == 3:
                poses_dict[key] = poses_dict[key].reshape(bs, -1)

    poses = None
    for key in ['global_orient', 'body_pose', 'jaw_pose', 'leye_pose', 'reye_pose', 'left_hand_pose', 'right_hand_pose']:
        if key in poses_dict.keys():
            if poses is None:
                poses = poses_dict[key]
            else:
                poses = torch.hstack((
                    poses,
                    poses_dict[key]
                ))
    if poses is None:
        # check if supr or star
        poses = poses_dict.get('pose', None)
    if poses is None:
        raise ValueError(f"Did not find any usable pose information in poses_dict with keys: {list(poses_dict.keys())}")
    return poses


def zero_from_joint_index(model_type: BodyModelType,
                          pose: torch.Tensor,
                          joint_idx: int,
                          pose_representation: PoseRepresentation
                          ) -> torch.Tensor:
    r"""Zeroes all joint parameters starting from the given joint index"""
    if len(pose.shape) == 1:
        pose = pose.unsqueeze(0)
        batch_dim = False
    else:
        batch_dim = True
    split_point = joint_idx * pose_representation.get_number_components()
    total_number_pose_params = MODEL_STATS[model_type].joints * pose_representation.get_number_components()
    remaining = total_number_pose_params - split_point
    zeroed = torch.cat(
        (pose[:, :split_point], torch.zeros(len(pose), remaining)),
        dim=1
    )
    if not batch_dim:
        zeroed = zeroed.squeeze(0)
    return zeroed


def direct_transfer_pose_parameters(model_from: BodyModelType,
                                    model_to: BodyModelType,
                                    current_pose_rep: PoseRepresentation,
                                    transfer_pose_rep: PoseRepresentation,
                                    poses: torch.Tensor
                                    ) -> torch.Tensor:
    """
        Directly transfers the given pose parameters from body model type model_from to body model type model_to.

        Params
        ------
            model_from (BodyModelType):
                Body model type from which the pose parameters are taken.
            model_to (BodyModelType):
                Body model type to which the pose parameters should be transferred.
            current_pose_rep (PoseRepresentation):
                Pose representation in which the poses argument is provided.
            transfer_pose_rep (PoseRepresentation):
                Pose representation in which the transfer should be performed.
            poses (torch.Tensor):
                Poses of shape (B, N) where B gives the batch size and N corresponds to the number of pose parameters
                according to the current_pose_rep argument: N = n_joints_input * current_pose_rep components

        Returns
        -------
            torch.Tensor:
                The pose parameters transferred to body model type model_to in pose representation transfer_pose_rep.
                Shape is (B, M) where usually, M != N with M = n_joints_target * transfer_pose_rep components
    """
    # 1. Determine and load module that converts current_pose_rep to transfer_pose_rep
    pose_converter = get_rotation_representation_conversion(current_pose_rep, transfer_pose_rep)
    # 2. transform poses using this module
    conv_poses = pose_converter(poses)
    # 3. Perform transfer with transformed poses data
    transf_indices = direct_transfer_indices(model_from, model_to, transfer_pose_rep)
    transferred_poses = conv_poses[:, transf_indices]
    pose_elems = transfer_pose_rep.get_number_components()

    if model_from in [BodyModelType.SMPL, BodyModelType.STAR]:
        # We don't need to do anything if we convert between SMPL and STAR
        if not ((model_from == BodyModelType.SMPL and model_to == BodyModelType.STAR) or (model_from == BodyModelType.STAR and model_to == BodyModelType.SMPL)):
            remaining = (MODEL_STATS[model_to]['joints'] * pose_elems) - (MODEL_STATS[model_from]['joints'] * pose_elems) + (2 * pose_elems) # left and right hand are not used
            transferred_poses = torch.cat(
                (
                    transferred_poses,
                    torch.zeros((len(transferred_poses), remaining),
                                dtype=torch.float32,
                                device=transferred_poses.device)
                ),
                dim=1
            )
    elif model_from == BodyModelType.SMPLH:
        if model_to in [BodyModelType.SMPL, BodyModelType.STAR]:
            # add zero hand poses
            transferred_poses = torch.cat(
                (
                    transferred_poses,
                    torch.zeros((len(transferred_poses), 2*pose_elems),
                                dtype=torch.float32,
                                device=transferred_poses.device)
                ),
                dim=1
            )
        elif model_to == BodyModelType.SMPLX:
            # add zero face poses
            transferred_poses = torch.cat(
                (
                    transferred_poses[:, :22*pose_elems],
                    torch.zeros((len(transferred_poses), 3*pose_elems),
                                dtype=torch.float32,
                                device=transferred_poses.device),
                    transferred_poses[:, 22*pose_elems:]
                ),
                dim=1
            )
        elif model_to == BodyModelType.SUPR:
            # add zero face and feet poses
            transferred_poses = torch.cat(
                (
                    transferred_poses[:, :22*pose_elems],
                    torch.zeros((len(transferred_poses), 3*pose_elems),
                                dtype=torch.float32,
                                device=transferred_poses.device),
                    transferred_poses[:, 22*pose_elems:],
                    torch.zeros((len(transferred_poses), 20*pose_elems),
                                dtype=torch.float32,
                                device=transferred_poses.device)
                ),
                dim=1
            )
    elif model_from == BodyModelType.SMPLX:
        if model_to in [BodyModelType.SMPL, BodyModelType.STAR]:
            # add zero hand poses
            transferred_poses = torch.cat(
                (
                    transferred_poses,
                    torch.zeros((len(transferred_poses), 2*pose_elems),
                                dtype=torch.float32,
                                device=transferred_poses.device)
                ),
                dim=1
            )
        elif model_to == BodyModelType.SUPR:
            # add zero feet poses
            transferred_poses = torch.cat(
                (
                    transferred_poses,
                    torch.zeros((len(transferred_poses), 20*pose_elems),
                                dtype=torch.float32,
                                device=transferred_poses.device)
                ),
                dim=1
            )
    elif model_from == BodyModelType.SUPR:
        if model_to in [BodyModelType.SMPL, BodyModelType.STAR]:
            # add zero hand poses
            transferred_poses = torch.cat(
                (
                    transferred_poses,
                    torch.zeros((len(transferred_poses), 2*pose_elems),
                                dtype=torch.float32,
                                device=transferred_poses.device)
                ),
                dim=1
            )
    return transferred_poses


def batched_rotation_parameter_conversion(poses: torch.Tensor,
                                          convertor: PoseRepresentationConverter,
                                          batch_size: int
                                          ) -> torch.Tensor:
    """
        Performs batched conversion of pose parameters.

        Params
        ------
            poses (torch.Tensor):
                Pose parameters that should be converted of shape (B, N)
            convertor (PoseRepresentationConverter):
                Class that performs the conversion
            batch_size (int):
                Batch size. -1 for converting all parameters at once

        Returns
        -------
            torch.Tensor: Converted pose parameters of shape (B, M)
    """
    bs = len(poses) if batch_size <= 0 else batch_size
    poses_conv = None
    for i in range(0, len(poses), bs):
        if poses_conv is None:
            poses_conv = convertor(poses[i:i+bs])
        else:
            poses_conv = torch.vstack((poses_conv, convertor(poses[i:i+bs])))
    return poses_conv


def adapt_number_shape_parameters(shape: torch.Tensor,
                                  desired_number: int
                                  ) -> torch.Tensor:
    """
        Adapts the number of shape parameters in the given tensor to match the desired number
        either by cutting later elements or by padding with zeros.

        Params
        ------
            shape (torch.Tensor):
                Shape parameters of shape (B, N)
            desired_number (int):
                Desired number of shape parameters

        Returns
        -------
            torch.Tensor: Adapted shape parameters of shape (B, desired_number)
    """
    if shape.shape[1] >= desired_number:
        adapted = shape[:, :desired_number]
    else:
        adapted = torch.hstack((
            shape,
            torch.zeros((len(shape), desired_number - shape.shape[1]), dtype=shape.dtype, device=shape.device)
        ))
    return adapted
