# Copyright (c) 2025 Julien Fischer, Chair of Computer Graphics and Visualization, TUD Dresden University of Technology.
# All rights reserved.
# Licensed under the MIT License. See LICENSE in the project root for license information.

import vedo
import torch
import numpy as np

from dotmap import DotMap
from typing import List, Optional, Union
from smpl_conversion.utils.model_mappings import PARENT_JOINT_MAPPINGS
from smpl_conversion.utils.model_infos import MODEL_STATS, MODELS
from smpl_conversion.utils.enum_configurations import BodyModelType


def get_bones_for_joints(model_type: BodyModelType,
                         joints: np.ndarray,
                         **kwargs
                         ) -> List[vedo.Line]:
    r"""Returns the bones between the given joints for the given body model"""
    assert model_type in MODELS, f"Unsupported model type: {model_type}"
    bones = [
        vedo.Line(joints[joint],
                  joints[PARENT_JOINT_MAPPINGS[model_type][joint]],
                  **kwargs
                  )
        for joint in range(int(MODEL_STATS[model_type].pose_params/3)) if PARENT_JOINT_MAPPINGS[model_type][joint] is not None
    ]
    return bones


def get_drawable_human(model_type: BodyModelType,
                       model: torch.nn.Module,
                       betas: torch.Tensor,
                       pose: DotMap[str, torch.Tensor],
                       transl: torch.Tensor,
                       joint_regressor: Optional[torch.Tensor] = None,
                       drawing_offset: Optional[np.ndarray] = None,
                       mesh_color: str = 'green7',
                       body_mesh_opacity: float = 1.0,
                       draw_joints: bool = True,
                       draw_bones: bool = True,
                       joint_color: str = 'red',
                       bone_color: str = 'blue',
                       joint_radius: int = 10,
                       bone_linewidth: int = 2,
                       add_flagpost: bool = True,
                       flagpost_caption: str = None,
                       return_verts: bool = False,
                       return_joints: bool = False,
                       ) -> DotMap:
    """
        Creates drawable vedo objects accessible via .body_mesh ; .bones ; .joints
        Optionally, the DotMap can also contain joints from the provided joint regressor (.joints_regressor),
        meshes for the joints (.joint_meshes), meshes for the bones (.bone_meshes), a flagpost (.flagpost),
        and the raw vertices (.raw_vertices).
    """
    if drawing_offset is None:
        drawing_offset = np.zeros(3, dtype=np.float32)
    if model_type in [BodyModelType.SUPR, BodyModelType.STAR]:
        model_output = model(**pose, betas=betas, trans=transl)
    else:
        model_output = model(betas=betas, **pose, transl=transl)
    ret = DotMap(_dynamic=False)
    # Get vertices
    if model_type in [BodyModelType.SUPR, BodyModelType.STAR]:
        verts = model_output[0].detach().cpu()
    else:
        verts = model_output.vertices[0].detach().cpu()
    if return_verts:
        ret.raw_vertices = verts
    # Get joints
    if joint_regressor is not None:
        joints_reg = joint_regressor @ verts
    else:
        joints_reg = None
    if model_type in [BodyModelType.SUPR, BodyModelType.STAR]:
        joints = model_output.J_transformed[0, :int(MODEL_STATS[model_type].pose_params/3)].detach().cpu()
    else:
        joints = model_output.joints[0, :int(MODEL_STATS[model_type].pose_params/3)].detach().cpu()
    if return_joints:
        ret.joints_regressor = joints_reg
        ret.joints = joints
    body_mesh = vedo.Mesh([verts + drawing_offset, model.faces]).c(mesh_color).opacity(body_mesh_opacity)
    ret.body_mesh = body_mesh
    if draw_joints:
        joint_points = vedo.Points(joints + drawing_offset, r=joint_radius).c(joint_color)
        ret.joint_meshes = joint_points
    if draw_bones:
        bones = get_bones_for_joints(model_type, joint_points.points(), c=bone_color, lw=bone_linewidth)
        ret.bone_meshes = bones
    if add_flagpost:
        fp = body_mesh.flagpost(txt=flagpost_caption, c=mesh_color)
        ret.flagpost = fp
    return ret

