# Copyright (c) 2025 Julien Fischer, Chair of Computer Graphics and Visualization, TUD Dresden University of Technology.
# All rights reserved.
# Licensed under the MIT License. See LICENSE in the project root for license information.

import torch
import numpy as np
from typing import Union
from dotmap import DotMap
from smpl_conversion.utils.enum_configurations import PoseRepresentation, BodyModelType

MODELS = BodyModelType.get_supported_body_model_types()#['smpl', 'smplh', 'smplx', 'supr', 'star']

MODEL_STATS = DotMap({
    # 'smpl': {
    #     'verts': 6890,
    #     'faces': 13776,
    #     'joints': 24,
    #     'pose_params': 72
    # },
    # 'smplh': {
    #     'verts': 6890,
    #     'faces': 13776,
    #     'joints': 52,
    #     'pose_params': 156
    # },
    # 'smplx': {
    #     'verts': 10475,
    #     'faces': 20908,
    #     'joints': 55,
    #     'pose_params': 165
    # },
    # 'supr': {
    #     'verts': 10475,
    #     'faces': 20908,
    #     'joints': 75,
    #     'pose_params': 225
    # },
    # 'star': {
    #     'verts': 6890,
    #     'faces': 13776,
    #     'jonts': 24,
    #     'pose_params': 72
    # },
    BodyModelType.SMPL: {
        'verts': 6890,
        'faces': 13776,
        'joints': 24,
        'pose_params': 72
    },
    BodyModelType.SMPLH: {
        'verts': 6890,
        'faces': 13776,
        'joints': 52,
        'pose_params': 156
    },
    BodyModelType.SMPLX: {
        'verts': 10475,
        'faces': 20908,
        'joints': 55,
        'pose_params': 165
    },
    BodyModelType.SUPR: {
        'verts': 10475,
        'faces': 20908,
        'joints': 75,
        'pose_params': 225
    },
    BodyModelType.STAR: {
        'verts': 6890,
        'faces': 13776,
        'joints': 24,
        'pose_params': 72
    }
},
_dynamic=False)

# FROM model_type TO model_type
CORRESPONDENCE_FNAMES = DotMap({
    # 'smpl': {
    #     'smplh': "smpl2smplh_def_transfer.pkl",
    #     'smplx': "smpl2smplx_deftrafo_setup.pkl",
    #     'supr': "smpl2smplx_deftrafo_setup.pkl"
    # },
    # 'smplh': {
    #     'smpl': "smplh2smpl_def_transfer.pkl",
    #     'smplx': "smplh2smplx_deftrafo_setup.pkl",
    #     'supr': "smplh2smplx_deftrafo_setup.pkl"
    # },
    # 'smplx': {
    #     'smpl': "smplx2smpl_deftrafo_setup.pkl",
    #     'smplh': "smplx2smplh_deftrafo_setup.pkl",
    #     'supr': None
    # },
    # 'supr': {
    #     'smpl': "smplx2smpl_deftrafo_setup.pkl",
    #     'smplh': "smplx2smplh_deftrafo_setup.pkl",
    #     'smplx': None
    # },
    # 'star': {
    #     'smplh': "smpl2smplh_def_transfer.pkl",
    #     'smplx': "smpl2smplx_deftrafo_setup.pkl",
    #     'supr': "smpl2smplx_deftrafo_setup.pkl"
    # },
    BodyModelType.SMPL: {
        BodyModelType.SMPLH: "smpl2smplh_def_transfer.pkl",
        BodyModelType.SMPLX: "smpl2smplx_deftrafo_setup.pkl",
        BodyModelType.SUPR: "smpl2smplx_deftrafo_setup.pkl",
        BodyModelType.STAR: None
    },
    BodyModelType.SMPLH: {
        BodyModelType.SMPL: "smplh2smpl_def_transfer.pkl",
        BodyModelType.SMPLX: "smplh2smplx_deftrafo_setup.pkl",
        BodyModelType.SUPR: "smplh2smplx_deftrafo_setup.pkl",
        BodyModelType.STAR: "smplh2smpl_def_transfer.pkl"
    },
    BodyModelType.SMPLX: {
        BodyModelType.SMPL: "smplx2smpl_deftrafo_setup.pkl",
        BodyModelType.SMPLH: "smplx2smplh_deftrafo_setup.pkl",
        BodyModelType.SUPR: None,
        BodyModelType.STAR: "smplx2smpl_deftrafo_setup.pkl"
    },
    BodyModelType.SUPR: {
        BodyModelType.SMPL: "smplx2smpl_deftrafo_setup.pkl",
        BodyModelType.SMPLH: "smplx2smplh_deftrafo_setup.pkl",
        BodyModelType.SMPLX: None,
        BodyModelType.STAR: "smplx2smpl_deftrafo_setup.pkl"
    },
    BodyModelType.STAR: {
        BodyModelType.SMPL: None,
        BodyModelType.SMPLH: "smpl2smplh_def_transfer.pkl",
        BodyModelType.SMPLX: "smpl2smplx_deftrafo_setup.pkl",
        BodyModelType.SUPR: "smpl2smplx_deftrafo_setup.pkl"
    },
    'smplx2smpl_special': "smplx_to_smpl.pkl",
    'smplx_mask_ids': "smplx_mask_ids.npy"
},
_dynamic=False)


def determine_model_from_pose(pose: Union[torch.Tensor, np.ndarray, int],
                              rotation_representation: PoseRepresentation
                              ) -> Union[BodyModelType, None]:
    if not isinstance(pose, int):
        if pose.ndim == 1:
            pose = pose.shape[0]
        else:
            pose = pose.shape[1]
    for model in MODEL_STATS.keys():
        if pose == MODEL_STATS[model]['joints'] * rotation_representation.get_number_components():
            return model
    return None
