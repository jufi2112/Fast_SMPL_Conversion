# Copyright (c) 2025 Julien Fischer, Chair of Computer Graphics and Visualization, TUD Dresden University of Technology.
# All rights reserved.
# Licensed under the MIT License. See LICENSE in the project root for license information.

import torch
import numpy as np
from typing import Union, Tuple
from dotmap import DotMap
from smpl_conversion.utils.enum_configurations import BodyModelType
from smpl_conversion.utils.model_infos import MODEL_STATS
from smpl_conversion.utils.Quaternions.quat import Quat

def parse_motion_x_322_file(x: Union[np.ndarray, torch.Tensor],
                            n_shape_components: int,
                            return_split: bool,
                            subsample_step_size: int = 1
                            ) -> Union[Union[np.ndarray, torch.Tensor],
                                       DotMap[str, Union[np.ndarray, torch.Tensor]]]:
    """
        Parses the given Motion-X 322-parameter pose description into translation,
        shape, and pose parameters. Returns either a single array/tensor in our
        common format [trans, shape, pose] or a DotMap of splitted arrays/tensors.

        Params
        ------
            x (np.ndarray or torch.Tensor):
                The motion-x data of shape (B, 322)
            n_shape_components (int):
                The number of shape components the returned shape parameters should have
            return_split (bool):
                If True, returns a DotMap of splitted 'trans', 'betas', and 'poses'
                data. If False, returns horizontally concantenated data
                as [trans, betas, poses]
            subsample_step_size (int):
                Subsampling step size. Defaults to 1

        Note
        ----
            Does NOT perform any coordinate system transformations
    """
    # Motion-X 322-parameter structure:
    # 3 - Global Orientation
    # 21*3 - Body Pose
    # 3 - Jaw
    # !!! no left and right eye parameters !!!
    # 30*3 - Left and Right Hand
    # 50 - Expression
    # 100 - Face Shape
    # 3 - Translation
    # 10 - Shape
    is_np = isinstance(x, np.ndarray)
    assert x.shape[1] == 322, f"Expected array to have 322 parameters, got {x.shape[1]}"
    trans = x[::subsample_step_size, 309:312]
    betas = x[::subsample_step_size, 312:]
    bs = trans.shape[0]
    if betas.shape[1] < n_shape_components:
        if is_np:
            betas = np.hstack((betas, np.zeros((bs, n_shape_components - betas.shape[1]), dtype=betas.dtype)))
        else:
             betas = torch.hstack((betas, torch.zeros((bs, n_shape_components - betas.shape[1]),
                                                      dtype=betas.dtype,
                                                      device=betas.device)))
    elif betas.shape[1] > n_shape_components:
        betas = betas[..., :n_shape_components]
    if is_np:
        poses = np.hstack((
            x[::subsample_step_size, :69],
            np.zeros((bs, 6), dtype=x.dtype),
            x[::subsample_step_size, 69:159]
        ))
    else:
         poses = torch.hstack((
             x[::subsample_step_size, :69],
             torch.zeros((bs, 6), dtype=x.dtype, device=x.device),
             x[::subsample_step_size, 69:159]
        ))
    assert poses.shape[1] == MODEL_STATS[BodyModelType.SMPLX]['pose_params']
    if return_split:
        return DotMap({
             'trans': trans,
             'betas': betas,
             'poses': poses
        }, _dynamic=False)
    if is_np:
        return np.hstack((trans, betas, poses))
    else:
        return torch.hstack((trans, betas, poses))


def align_motion_x_to_amass(global_orientations: np.ndarray,
                            translation: np.ndarray
                            ) -> Tuple[np.ndarray, np.ndarray]:
    """
        Aligns the given Motion-X global orientation and translation parameters (OpenGL coordinate system)
        to the AMASS reference system (Blender coordinate system). Note that due
        to the difference between pivot point (pelvis joint) and local coordinate system origin (somewhere in the chest),
        the resulting SMPL-X meshes will not be aligned to the ground floor (x-y plane)

        Params
        ------
            global_orientation (np.ndarray):
                Global orientation parameters of shape (B, 3)
            translation (np.ndarray):
                Translation parameters of shape (B, 3)

        Returns
        -------
            np.ndarray:
                The aligned global orientation parameters of shape (B, 3)
            np.ndarray:
                The aligned translation parameters of shape (B, 3)
    """
    # OpenGL to Blender requires to rotate by 90 degrees around x-Axis
    transf_quat = Quat.from_axis_and_angle(np.asarray([1,0,0]), 0.5*np.pi)
    transf_mat = np.asarray([
        [1,0,0],
        [0,0,-1],
        [0,1,0]
    ], dtype=np.float32)
    go = Quat.to_rot_vec(transf_quat @ Quat.from_rot_vec(global_orientations, return_raw_numpy=True)).reshape(-1, 3)
    bs = len(translation)
    transl = (transf_mat @ translation.reshape(bs, 3, 1)).reshape(bs, 3)
    return go, transl
