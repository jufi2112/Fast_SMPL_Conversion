# Copyright (c) 2025 Julien Fischer, Chair of Computer Graphics and Visualization, TUD Dresden University of Technology.
# All rights reserved.
# Licensed under the MIT License. See LICENSE in the project root for license information.

import torch
import smplx

import numpy as np
import torch.nn as nn
import os.path as osp

from dotmap import DotMap
from typing import Union, Tuple
from supr.pytorch.supr import SUPR
from smpl_conversion.utils.model_infos import MODEL_STATS
from smpl_conversion.utils.model_correspondences import get_correspondence_matrix
from smpl_conversion.utils.mpi_code.mpi_edge_loss import get_vertices_per_edge, compute_edges
from smpl_conversion.models.combined import split_network_output
from smpl_conversion.utils.enum_configurations import BodyModelType, PoseRepresentation
from smpl_conversion.factories import BodyModelFactory
from smpl_conversion.utils.model_evaluation import get_model_vertices_fast
from smpl_conversion.data.transforms import get_rotation_representation_conversion, PoseRepresentationConverter

class CorrespondenceLoss(nn.Module):
    def __init__(self,
                input_model_type: BodyModelType,
                target_model_type: BodyModelType,
                gender: str,
                model_location: str,
                transfer_file_location: str,
                expected_batch_size: int,
                num_betas: int,
                device: Union[str, torch.device],
                supr_constrained: bool,
                input_rotation_representation: PoseRepresentation,
                prediction_rotation_representation: PoseRepresentation,
                verbosity: int = 0,
                epsilon: float = 1.e-8
                ):
        """
            A loss that is calculated based on the vertex and edge correspondences of the input and target body model
            types. By using these correspondences, no ground truth parameters are required for the target body model type.

            Params
            ------
                input_model_type (BodyModelType):
                    Input body model type.
                target_model_type (BodyModelType):
                    Target body model type.
                gender (str):
                    Gender of the body models. Same gender is used for input and target.
                model_location (str):
                    Directory that contains the body models.
                transfer_file_location (str):
                    Directory that contains the vertex correspondences. The correct file is automatically
                    inferred from the input and target model types.
                expected_batch_size (int):
                    Batch size for the input and target models that should be constant for most
                    of the training (excluding e.g. the last batch of the epoch).
                num_betas (int):
                    Number of shape components. Will be used for both input and target models.
                device (str or torch.device):
                    Device where the loss should be calculated on.
                supr_constrained (bool):
                    Whether the SUPR model (if one is used) should be constrained.
                    Based on this (and the gender), the correct SUPR model is loaded.
                input_rotation_representation (PoseRepresentation):
                    Format in which the input rotation parameters will be given.
                    Is used to determine whether a model based on rotation vectors
                    or rotation matrices should be used.
                prediction_rotation_representation (PoseRepresentation):
                    Format in which the predicted rotation parameters will be given.
                    Is used to determine whether a model based on rotation
                    vectors or rotation matrices should be created.
                verbosity (int):
                    Verbosity setting
                epsilon (float):
                    Epsilon that is added before square root calculation. Defaults to 1.e-8
        """
        super().__init__()
        corr_mat = get_correspondence_matrix(transfer_file_location,
                                             input_model_type,
                                             target_model_type)
        corr_mat = corr_mat.to(device)
        self.register_buffer('corr', corr_mat)
        self.model_from = input_model_type
        self.model_to = target_model_type
        self.device = device
        self.num_betas = num_betas
        self.input_model_params = None
        self.input_model = None
        self.target_model_params = None
        self.target_model = None
        self.verbosity = verbosity
        self.supr_constrained = supr_constrained
        self.input_rotation_representation = input_rotation_representation
        self.prediction_rotation_representation = prediction_rotation_representation
        # If input pose parameters are rotation vectors, we use a body model based on rotation vectors (input_rot_repr_conv will be noop)
        # If input pose parameters are something else, we use a body model based on rotation matrix (input_rot_repr_conv will convert from input rotation representation to rotation matrix)
        # Same for prediction results
        # ----
        # These convertors are for converting FROM THE NETWORK's OUTPUT FORMAT TO WHAT THE BODY MODEL EXPECTS (rotation vector or rotation matrix)
        self.input_rot_repr_conv: PoseRepresentationConverter = get_rotation_representation_conversion(self.input_rotation_representation,
                                                                                                       PoseRepresentation.ROTATION_VECTOR if self.input_rotation_representation == PoseRepresentation.ROTATION_VECTOR else PoseRepresentation.ROTATION_MATRIX_9D)
        self.pred_rot_repr_conv: PoseRepresentationConverter = get_rotation_representation_conversion(self.prediction_rotation_representation,
                                                                                                      PoseRepresentation.ROTATION_VECTOR if self.prediction_rotation_representation == PoseRepresentation.ROTATION_VECTOR else PoseRepresentation.ROTATION_MATRIX_9D)
        self.gender = gender
        self.body_model_location = model_location
        self.epsilon = epsilon
        self.input_model, self.input_model_params = BodyModelFactory.create_body_model(self.model_from,
                                                                                       self.gender,
                                                                                       self.num_betas,
                                                                                       model_location,
                                                                                       self.device,
                                                                                       self.input_rotation_representation != PoseRepresentation.ROTATION_VECTOR,
                                                                                       expected_batch_size=expected_batch_size,
                                                                                       supr_is_constrained=self.supr_constrained
                                                                                       )
        self.target_model, self.target_model_params = BodyModelFactory.create_body_model(self.model_to,
                                                                                         self.gender,
                                                                                         self.num_betas,
                                                                                         model_location,
                                                                                         self.device,
                                                                                         self.prediction_rotation_representation != PoseRepresentation.ROTATION_VECTOR,
                                                                                         expected_batch_size=expected_batch_size,
                                                                                         supr_is_constrained=self.supr_constrained
                                                                                         )
        self.mask_ids = self._load_mask_ids(transfer_file_location)
        self.mask_ids_copy = self.mask_ids.copy() if isinstance(self.mask_ids, np.ndarray) else None
        if self.mask_ids is not None and self.verbosity > 0:
            print("CorrespondenceLoss: Will use mask_ids for conversion")
        if self.verbosity > 0:
            if self.input_rotation_representation == PoseRepresentation.ROTATION_VECTOR:
                print("Will use rotation vector-based body model for input-type model")
            else:
                print("Will use rotation matrix-based body model for input-type model")
            if self.prediction_rotation_representation == PoseRepresentation.ROTATION_VECTOR:
                print("Will use rotation vector-based body model for prediction-type model.")
            else:
                print("Will use rotation matrix-based body model for prediction-type model.")
        # Prepare data for edge loss
        target_faces = self.target_model.faces
        if isinstance(target_faces, torch.Tensor):
            target_faces = target_faces.detach().cpu().numpy()
        target_n_vertices = MODEL_STATS[self.model_to].verts
        if self.mask_ids is None:
            f_sel = np.ones_like(target_faces[:, 0], dtype=np.bool_)
        else:
            f_per_v = [[] for _ in range(target_n_vertices)]
            [f_per_v[vv].append(iff) for iff, ff in enumerate(target_faces) for vv in ff]
            f_sel = list(set(tuple(sum([f_per_v[vv] for vv in self.mask_ids], []))))
        vpe = get_vertices_per_edge(MODEL_STATS[self.model_to].verts,
                                    target_faces[f_sel])
        vpe = torch.tensor(vpe, device=self.device)
        self.register_buffer('target_vertices_per_edge', vpe)


    def _load_mask_ids(self, transfer_file_location: str) -> np.ndarray:
        if self.model_to not in [BodyModelType.SMPLX, BodyModelType.SUPR]:
            return None
        if self.model_from not in [BodyModelType.SMPL, BodyModelType.SMPLH, BodyModelType.STAR]:
            return None
        fpath = osp.join(transfer_file_location, 'smplx_mask_ids.npy')
        if not osp.exists(fpath):
            raise FileNotFoundError(
                f"Could not find file that contains mask ids at {fpath}"
            )
        mask_ids = np.load(fpath)
        return mask_ids


    def _update_models_batch_size(self,
                                  batch_size: int
                                  ) -> None:
        if not hasattr(self, 'input_model') or not hasattr(self, 'target_model'):
            raise ValueError("Have to create models before updating them!")
        if self.model_from not in [BodyModelType.SUPR, BodyModelType.STAR]:
            if not self.input_model_params.batch_size == batch_size:
                self.input_model, self.input_model_params = BodyModelFactory.create_body_model(self.model_from,
                                                                                               self.gender,
                                                                                               self.num_betas,
                                                                                               self.body_model_location,
                                                                                               self.device,
                                                                                               self.input_rotation_representation != PoseRepresentation.ROTATION_VECTOR,
                                                                                               expected_batch_size=batch_size,
                                                                                               supr_is_constrained=self.supr_constrained
                                                                                               )
        if self.model_to not in [BodyModelType.SUPR, BodyModelType.STAR]:
            if not self.target_model_params.batch_size == batch_size:
                self.target_model, self.target_model_params = BodyModelFactory.create_body_model(self.model_to,
                                                                                                 self.gender,
                                                                                                 self.num_betas,
                                                                                                 self.body_model_location,
                                                                                                 self.device,
                                                                                                 self.prediction_rotation_representation != PoseRepresentation.ROTATION_VECTOR,
                                                                                                 expected_batch_size=batch_size,
                                                                                                 supr_is_constrained=self.supr_constrained
                                                                                                 )


    def forward(self,
                input_parameters: Union[DotMap, torch.Tensor],
                predicted_parameters: Union[DotMap, torch.Tensor],
                loss_type: str = 'vertex',
                reduction_mode: str = 'mean',
                return_per_vertex_loss: bool = False,
                **kwargs
                ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, Union[None, np.ndarray]]]:
        """
            Calculates the L2 loss between the input model and predicted target
            model based on the given parameters.

            Params
            ------
                input_parameters (DotMap or torch.Tensor):
                    Input model type parameters with keys 'betas', 'poses', 'trans' or
                    tensor that contains these parameters horizontally stacked, i.e. [trans, betas, poses].
                    Expected to be in the rotation representation provided to the constructor
                predicted_parameters (DotMap or torch.Tensor):
                    Predicted model type parameters with keys 'betas', 'poses', 'trans'
                    or tensor that contains these parameters horzontally stacked, i.e. [trans, betas, poses].
                    Expected to be in the rotation representation provided to the constructor
                loss_type (str):
                    Loss type. Either 'vertex' (v2v) or 'edge' -based. Provide
                    'mse' to use vertex-based MSE loss from optimization approach. Defaults to 'vertex'
                reduction_mode (str):
                    'mean' or 'sum'. If 'sum', the result is summed over all vertices
                    of each mesh in the batch. Defaults to 'mean'
                return_per_vertex_loss (bool):
                    Whether, additionally to the reduced loss, a tensor containing the loss for
                    each vertex, as well as the ids of vertices used for the loss calculation,
                    should be returned.
                Accepts other kwargs for compatibility reasons, but does not use them.

            Returns
            -------
                torch.Tensor:
                    The loss as specified by the provided arguments.
                torch.Tensor:
                    If return_per_vertex_loss is True, a tensor that contains the loss for each vertex
                np.ndarray or None:
                    If return_per_vertex_loss is True, a numpy array that contains the
                    indices of valid vertices (the vertex indices with which the loss value was calculated).
                    If all vertices were used, None is returned
        """
        if isinstance(input_parameters, torch.Tensor):
            trans, shape, pose = split_network_output(input_parameters, self.num_betas)
            input_parameters = DotMap({'trans': trans, 'betas': shape, 'poses': pose}, _dynamic=False)
            del trans
            del shape
            del pose
        if isinstance(predicted_parameters, torch.Tensor):
            trans, shape, pose = split_network_output(predicted_parameters, self.num_betas)
            predicted_parameters = DotMap({'trans': trans, 'betas': shape, 'poses': pose}, _dynamic=False)
            del trans
            del shape
            del pose
        # Convert input and predicted poses from their representation to a representation that can be used by the body models (rotation vector or 3x3 rotation matrix)
        inp_conv_poses = self.input_rot_repr_conv(input_parameters['poses'])
        pred_conv_poses = self.pred_rot_repr_conv(predicted_parameters['poses'])
        bs_input = len(input_parameters[list(input_parameters.keys())[0]])
        bs_pred = len(predicted_parameters[list(predicted_parameters.keys())[0]])
        if bs_input != bs_pred:
            raise ValueError(f"Expected input and target parameters to have same batch size: {bs_input} vs {bs_pred}")
        if loss_type.lower() not in ['vertex', 'edge', 'mse']:
            raise ValueError(f"Loss type has to be 'vertex', 'edge', or 'mse', but got {loss_type}")
        batch_size = bs_input
        # Adapt models to batch size, if necessary
        self._update_models_batch_size(batch_size)
        # Evaluate models and get vertices
        input_vertices = get_model_vertices_fast(self.input_model,
                                                 PoseRepresentation.ROTATION_VECTOR if self.input_rotation_representation == PoseRepresentation.ROTATION_VECTOR else PoseRepresentation.ROTATION_MATRIX_9D,
                                                 {'trans': input_parameters['trans'],
                                                  'betas': input_parameters['betas'],
                                                  'poses': inp_conv_poses})
        predicted_vertices = get_model_vertices_fast(self.target_model,
                                                     PoseRepresentation.ROTATION_VECTOR if self.prediction_rotation_representation == PoseRepresentation.ROTATION_VECTOR else PoseRepresentation.ROTATION_MATRIX_9D,
                                                     {'trans': predicted_parameters['trans'],
                                                      'betas': predicted_parameters['betas'],
                                                      'poses': pred_conv_poses})

        gt_vertices = torch.einsum('mn,bnk->bmk', self.corr, input_vertices)

        if return_per_vertex_loss:
            with torch.no_grad():
                if loss_type.lower() == 'mse':
                    loss_per_vertex = (predicted_vertices - gt_vertices).pow(2).sum(dim=-1).mean(dim=0).cpu()
                elif loss_type.lower() == 'vertex':
                    loss_per_vertex = (predicted_vertices - gt_vertices).pow(2).sum(dim=-1).sqrt().mean(dim=0).cpu()
                else:
                    # Edge loss
                    loss_per_vertex = torch.zeros((predicted_vertices.shape[1]), device="cpu")

        # MSE loss from optimization approach
        if loss_type.lower() == 'mse':
            if self.mask_ids is not None:
                predicted_vertices = predicted_vertices[:, self.mask_ids]
                gt_vertices = gt_vertices[:, self.mask_ids]
            diff = predicted_vertices - gt_vertices
            loss = diff.pow(2).sum() / diff.shape[0]
            # = diff.pow(2).sum(dim=-1).mean()
            if return_per_vertex_loss:
                return loss, loss_per_vertex, self.mask_ids_copy
            else:
                return loss

        if loss_type.lower() == 'vertex':
            if self.mask_ids is not None:
                predicted_vertices = predicted_vertices[:, self.mask_ids]
                gt_vertices = gt_vertices[:, self.mask_ids]
            loss = predicted_vertices - gt_vertices
        elif loss_type.lower() == 'edge':
            gt_edges = compute_edges(gt_vertices, self.target_vertices_per_edge)
            pred_edges = compute_edges(predicted_vertices, self.target_vertices_per_edge)
            loss = gt_edges - pred_edges
        else:
            raise ValueError(f"Unsupported loss type: {loss_type.lower()}")

        # when using vertex_edge loss, this can become zero, which results in gradients of sqrt(0) becoming nan
        loss = torch.sqrt(loss.pow(2).sum(dim=-1) + self.epsilon)
        # loss = loss.pow(2).sum(dim=-1).sqrt()
        if reduction_mode == 'sum':
           loss = loss.sum()
        if return_per_vertex_loss:
            return loss.mean(), loss_per_vertex, self.mask_ids_copy
        else:
            return loss.mean()
