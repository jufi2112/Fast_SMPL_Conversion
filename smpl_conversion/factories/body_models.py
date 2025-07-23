# Copyright (c) 2025 Julien Fischer, Chair of Computer Graphics and Visualization, TUD Dresden University of Technology.
# All rights reserved.
# Licensed under the MIT License. See LICENSE in the project root for license information.

import torch
import smplx

from dotmap import DotMap
from os import path as osp
from supr.pytorch.supr import SUPR
from star.pytorch.star import STAR
from typing import Union, Tuple, Dict
from smpl_conversion.utils.enum_configurations import BodyModelType

class BodyModelFactory:
    """Creates body models."""
    @staticmethod
    def create_body_model(model_type: BodyModelType,
                          gender: str,
                          num_betas: int,
                          model_location: str,
                          device: Union[str, torch.device] = "cpu",
                          return_rot_mat_model: bool = False,
                          expected_batch_size: int = 1,
                          supr_is_constrained: bool = False
                          ) -> Tuple[
                                    Union[Union[smplx.SMPL, smplx.SMPLLayer],
                                          Union[smplx.SMPLH, smplx.SMPLHLayer],
                                          Union[smplx.SMPLX, smplx.SMPLXLayer],
                                          SUPR,
                                          STAR
                                          ],
                                    Dict
                                    ]:
        """
            Creates the specified body model with the specified arguments.

            Params
            ------
                model_type (BodyModelType):
                    The type of body model that should be created
                gender (str):
                    Gender of the body model.
                num_betas (int):
                    Number of shape parameters
                model_location (str):
                    Location of the body model files
                device (str or torch.device):
                    The device onto which the created model should be transferred.
                    Defaults to "cpu".
                return_rot_mat_model (bool):
                    For SMPL, SMPL+H, and SMPL-X body models: Whether the
                    models should support pose parameters in rotation matrix (True)
                    or rotation vector (False) format. Defaults to False
                expected_batch_size (int):
                    For SMPL, SMPL+H, and SMPL-X body models: The batch size for which
                    the models should be initialized. Defaults to 1
                supr_is_constrained (bool):
                    For SUPR body model: Whether the body model is constrained.
                    Defaults to False

            Returns
            -------
                torch.nn:
                    The requested body model, which is a subclass of torch.nn
                DotMap:
                    A dictionary-like that contains the parameters that were used to
                    create the body model

            Raises
            ------
                NotImplementedError:
                    If the requested body model type is not implemented
        """
        if model_type in [BodyModelType.SMPL, BodyModelType.SMPLH, BodyModelType.SMPLX]:
            # SMPL / SMPL+H / SMPL-X specific model initialization
            model_parameters = DotMap({
                'gender': gender,
                'model_path': model_location,
                'model_type': model_type.to_internal_string(),
                'flat_hand_mean': True,
                'use_pca': False,
                'batch_size': expected_batch_size,
                'ext': 'npz' if model_type == BodyModelType.SMPLH else 'pkl',
                'num_betas': num_betas
            }, _dynamic=False)
            if return_rot_mat_model:
                model = smplx.build_layer(**model_parameters)
            else:
                model = smplx.create(**model_parameters)
        elif model_type == BodyModelType.SUPR:
            model_parameters = DotMap({
                'path_model': osp.join(model_location,
                                       'supr',
                                       f"supr_{gender.lower()}{'_constrained.npy' if supr_is_constrained else '.npy'}"
                                       ),
                'num_betas': num_betas,
                'constrained': supr_is_constrained
            }, _dynamic=False)
            if return_rot_mat_model:
                print("Warning: SUPR does not implement rotation matrix-based model. Returning rotation vector-based model.")
            model = SUPR(**model_parameters)
        elif model_type == BodyModelType.STAR:
            model_parameters = DotMap({
                'model_path': osp.join(model_location,
                                       'star',
                                       f"star_{gender.lower()}.npz"
                                       ),
                'num_betas': num_betas
            }, _dynamic=False)
            if return_rot_mat_model:
                print("Warning: STAR does not implement rotation matrix-based model. Returning rotation vector-based model.")
            model = STAR(**model_parameters)
        else:
            raise NotImplementedError(f"Model type {model_type.to_string()} is not supported")
        model = model.to(device)
        return model, model_parameters
