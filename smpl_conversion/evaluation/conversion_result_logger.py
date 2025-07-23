# Copyright (c) 2025 Julien Fischer, Chair of Computer Graphics and Visualization, TUD Dresden University of Technology.
# All rights reserved.
# Licensed under the MIT License. See LICENSE in the project root for license information.

import os
import torch
from os import path as osp
from typing import Union, Dict, List
from smpl_conversion.utils.enum_configurations import (
    BodyModelType,
    PoseRepresentation
)
from smpl_conversion.utils.parameter_processing import direct_transfer_pose_parameters
from smpl_conversion.utils.model_infos import MODEL_STATS
from smpl_conversion.utils.model_mappings import JOINT_NAMES

class ConversionResultLogger:
    def __init__(self,
                 fpath: str,
                 model_from: BodyModelType,
                 model_to: BodyModelType,
                 rotation_representation: PoseRepresentation,
                 n_shape_components: int,
                 enabled: bool,
                 log_translation: bool = True,
                 shape_components_to_log: int = -1,
                 joints_to_log: List[int] = None
                 ):
        """
            Logs the conversion results to the given file.

            Params
            ------
                fpath (str):
                    File to which the results should be logged
                model_from (BodyModelType):
                    Model from which conversion is performed
                model_to (BodyModelType):
                    Model to which conversion is performed
                rotation_representation (PoseRepresentation):
                    Rotation representation
                n_shape_components (int):
                    Number of shape components
                enabled (bool):
                    Whether the logger should log data. Can be used to only
                    conditionally use the logger
                log_translation (bool):
                    Whether translation parameters should be logged. Defaults
                    to True.
                shape_components_to_log (int):
                    How many shape parameters should be logged (starting from
                    the first). Provide 0 to disable logging of shape
                    parameters. Defaults to -1 (log all)
                joints_to_log (list of int):
                    A list of joint indices that should be logged.
                    Provide an empty list to not log any joints.
                    Defaults to None (log all joints)
        """
        self.file = None
        self.fpath = fpath
        self.model_from = model_from
        self.model_to = model_to
        self.rot_repr = rotation_representation
        self.n_shape_components = n_shape_components
        self.enabled = enabled
        if not self.enabled:
            return
        self.log_translation = log_translation
        if shape_components_to_log == -1 or shape_components_to_log > self.n_shape_components:
            self.betas_to_log = [x for x in range(self.n_shape_components)]
        else:
            self.betas_to_log = [x for x in range(shape_components_to_log)]
        if joints_to_log is None:
            self.joints_to_log = [x for x in range(MODEL_STATS[self.model_to])]
        else:
            self.joints_to_log = joints_to_log
        dir = osp.dirname(self.fpath)
        if not osp.exists(dir):
            os.makedirs(dir)
        if osp.exists(self.fpath):
            raise ValueError(f"Provided file {self.fpath} already exists!")


    def __enter__(self):
        if self.enabled:
            self.file = open(self.fpath, 'a')
            self._write_header()
        return self


    def __exit__(self, exc_type, exc_value, traceback):
        if self.file:
            self.file.write('\n')
            self.file.close()


    def log(self,
            input: Union[Dict[str, torch.Tensor], torch.Tensor],
            pred: Union[Dict[str, torch.Tensor], torch.Tensor]
            ):
        """
            Logs the results to file.

            Params
            ------
                input (torch.Tensor or Dict):
                    Input parameters of shape (B, 3 + n_shape_components + n_pose_params_input_model).
                    Input pose parameters are automatically directly transferred to target body model type.
                    If a dict, expected keys are 'trans', 'betas', 'poses'
                pred (torch.Tensor or Dict):
                    Predicted parameters of shape (B, 3 + n_shape_components + n_pose_params_output_model)
                    If a dict, expected keys are 'trans', 'betas', 'poses'
        """
        if not self.enabled:
            return
        if self.file is None:
            raise RuntimeError("Log file is not open. Use 'with Logger(...) as log:' context.")
        if isinstance(input, dict):
            inp_trans = input['trans']
            inp_betas = input['betas']
            inp_poses = input['poses']
        else:
            inp_trans = input[:, :3]
            inp_betas = input[:, 3:3+self.n_shape_components]
            inp_poses = input[:, 3+self.n_shape_components:]
        dir_transf_inp = direct_transfer_pose_parameters(self.model_from,
                                                         self.model_to,
                                                         self.rot_repr,
                                                         self.rot_repr,
                                                         inp_poses)
        inp = torch.hstack((inp_trans, inp_betas, dir_transf_inp))
        if isinstance(pred, dict):
            prediction = torch.hstack((pred['trans'], pred['betas'], pred['poses']))
        else:
            prediction = pred
        #assert inp.shape == prediction.shape, f"Incompatible shapes of direct transferred input {inp.shape} and predicted output {prediction.shape}"
        # The offset learned by the network
        offset = prediction - inp
        for idx in range(len(offset)):
            self._write_line(inp[idx], offset[idx])


    def _write_header(self):
        """
            Writes header information
        """
        if self.file is None:
            return
        # Format:
        # number of header lines
        # each header line contains the description of one column, ascending order
        rotation_descriptors = {
            PoseRepresentation.ROTATION_VECTOR: ['x', 'y', 'z'],
            PoseRepresentation.QUATERNION: ['q_a', 'q_b', 'q_c', 'q_d'],
            PoseRepresentation.ROTATION_MATRIX_9D: [f'm_{row+1}{col+1}' for row in range(3) for col in range(3)],
            PoseRepresentation.ROTATION_MATRIX_6D: [f'm_{row+1}{col+1}' for row in range(3) for col in range(2)]
        }
        input_trans_header = [
            translation_header for translation_header in ['i_t_x: input translation x component in m',
                                                          'i_t_y: input translation y component in m',
                                                          'i_t_z: input translation z component in m'] if self.log_translation
        ]
        output_trans_header = [
            translation_header for translation_header in ['o_t_x: learned offset translation x component in m',
                                                          'o_t_y: learned offset translation y component in m',
                                                          'o_t_z: learned offset translation z component in m'] if self.log_translation
        ]
        input_shape_header = [f'i_s_{idx+1}: input shape parameter {idx+1}' for idx in self.betas_to_log]
        output_shape_header = [f'o_s_{idx+1}: learned offset shape parameter {idx+1}' for idx in self.betas_to_log]
        input_pose_header = [
            f'i_j_{j_idx}_{axis}: input {str(self.rot_repr).lower()} {axis} component for joint {JOINT_NAMES[self.model_to][j_idx]}' for j_idx in self.joints_to_log for axis in rotation_descriptors[self.rot_repr]
        ]
        output_pose_header = [
            f'o_j_{j_idx}_{axis}: learned offset {str(self.rot_repr).lower()} {axis} component for joint {JOINT_NAMES[self.model_to][j_idx]}' for j_idx in self.joints_to_log for axis in rotation_descriptors[self.rot_repr]
        ]
        column_descriptions = [
            *input_trans_header,
            *input_shape_header,
            *input_pose_header,
            *output_trans_header,
            *output_shape_header,
            *output_pose_header
        ]
        self.file.write(f"{len(column_descriptions)}\n")
        for line in column_descriptions:
            self.file.write(line+"\n")


    def _write_line(self,
                    input_params: torch.Tensor,
                    offset: torch.Tensor):
        """
            Writes a single line of content

            Params
            ------
                input_params (torch.Tensor):
                    Input of shape (3 + n_shape_components + n_pose_parameters * n_components_rotation_representation)
                offset (torch.Tensor):
                    Learned offset tensor of same shape as input
        """
        if self.file is None:
            return
        # Format: input translation, shape, pose, learned offset translation, shape, pose
        inp_trans = input_params[:3]
        inp_betas = input_params[3:3+self.n_shape_components]
        inp_poses = input_params[3+self.n_shape_components:]
        # write input translation
        if self.log_translation:
            self.file.write(f'{inp_trans[0].item():.4f},{inp_trans[1].item():.4f},{inp_trans[2].item():.4f},') # Will always write predicted translation offset, so put comma
        # write input betas
        for idx in self.betas_to_log:
            self.file.write(f'{inp_betas[idx].item():.4f},') # Will always write predicted betas offset, so put comma
        # write input poses
        for idx in self.joints_to_log:
            self.file.write(f'{inp_poses[idx*3].item():.4f},{inp_poses[idx*3+1].item():.4f},{inp_poses[idx*3+2].item():.4f},')
        offset_trans = offset[:3]
        offset_betas = offset[3:3+self.n_shape_components]
        offset_poses = offset[3+self.n_shape_components:]
        # write predicted translation offset
        if self.log_translation:
            self.file.write(f'{offset_trans[0].item():.4f},{offset_trans[1].item():.4f},{offset_trans[2].item():.4f}{"," if len(self.betas_to_log) > 0 or len(self.joints_to_log) > 0 else ""}') # only put comma if we write another entry
        # write predicted betas offset
        for idx in self.betas_to_log:
            self.file.write(f'{offset_betas[idx].item():.4f}{"," if idx != self.betas_to_log[-1] or len(self.joints_to_log) > 0 else ""}')
        # write predicted poses offset
        for idx in self.joints_to_log:
            self.file.write(f'{offset_poses[idx*3].item():.4f},{offset_poses[idx*3+1].item():.4f},{offset_poses[idx*3+2].item():.4f}{"," if idx != self.joints_to_log[-1] else ""}')
        self.file.write("\n")
