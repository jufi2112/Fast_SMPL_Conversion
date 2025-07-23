# Copyright (c) 2025 Julien Fischer, Chair of Computer Graphics and Visualization, TUD Dresden University of Technology.
# All rights reserved.
# Licensed under the MIT License. See LICENSE in the project root for license information.

import torch
import numpy as np
import os
from os import path as osp
from .model_infos import CORRESPONDENCE_FNAMES, MODELS, MODEL_STATS
from .misc import get_existing_extension
from dotmap import DotMap
import pickle
from smpl_conversion.utils.enum_configurations import BodyModelType

def get_correspondence_matrices(transfer_files_location: str
                                ) -> DotMap[str, DotMap[str, torch.Tensor]]:
    conv_mats = DotMap({
        model_from: {
            model_to: None
            for model_to in MODELS
        }
        for model_from in MODELS
    })
    for model_from in MODELS:
        for model_to in MODELS:
            conv_mats[model_from][model_to] = get_correspondence_matrix(transfer_files_location,
                                                                        model_from,
                                                                        model_to)
    return conv_mats


def get_correspondence_matrix(transfer_files_location: str,
                              model_from: BodyModelType,
                              model_to: BodyModelType) -> torch.Tensor:
    if model_from == model_to or CORRESPONDENCE_FNAMES[model_from][model_to] is None:
        return torch.eye(MODEL_STATS[model_from]['verts'], dtype=torch.float32)
    with open(osp.join(transfer_files_location, CORRESPONDENCE_FNAMES[model_from][model_to]), 'rb') as file:
        content = pickle.load(file)
        if 'mtx' in content:
            mat = content['mtx']
        elif 'matrix' in content:
            mat = content['matrix']
        else:
            raise ValueError("The transfer file does not contain a valid key (mtx or matrix)")
    if not isinstance(mat, np.ndarray):
        mat = mat.toarray()
    n_verts = mat.shape[1] // 2
    mat = mat[:, :n_verts]
    mat = torch.tensor(mat, dtype=torch.float32)
    return mat


def get_joint_regressors(models_location: str,
                         gender: str,
                         ) -> DotMap[str, torch.Tensor]:
    regressors = DotMap({}, _dynamic=False)
    for model in MODELS:
        regressors[model] = get_joint_regressor(model, models_location, gender)
    return regressors


def get_joint_regressor(model_type: BodyModelType,
                        model_location: str,
                        gender: str,
                        ) -> torch.Tensor:
    assert model_type in MODELS, f"Unsupported model type: {model_type}"
    fname = f"{model_type.to_internal_string().upper()}_{gender.upper()}"
    ext = get_existing_extension(osp.join(model_location, model_type.to_internal_string()), fname, ['npz', 'pkl'])
    if ext is None:
        raise ValueError(f"Could not find a valid model file at {osp.join(model_location, model_type.to_internal_string())}")
    with open(osp.join(model_location, model_type.to_internal_string(), fname+ext), "rb") as file:
        if ext == '.npz':
            j_reg = np.load(file, allow_pickle=True)['J_regressor']
        else:
            j_reg = pickle.load(file)['J_regressor']
    if not isinstance(j_reg, np.ndarray):
        j_reg = j_reg.toarray()
    regressor = torch.tensor(j_reg, dtype=torch.float32)
    return regressor
