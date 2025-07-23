# Copyright (c) 2025 Julien Fischer, Chair of Computer Graphics and Visualization, TUD Dresden University of Technology.
# All rights reserved.
# Licensed under the MIT License. See LICENSE in the project root for license information.

# Portions of this file are adapted from PyTorch3D:
# https://github.com/facebookresearch/pytorch3d
# Licensed under the BSD 3-Clause License.

import torch
import numpy as np

from torch import nn
from dotmap import DotMap
from typing import Union, Tuple, List, Callable
from smpl_conversion.utils.enum_configurations import PoseRepresentation
from smpl_conversion.utils.mpi_code.mpi_pose_utils import batch_rot2aa, batch_rodrigues
from smpl_conversion.utils.math import batched_rot_mat_to_quat

class MinMaxScaler():
    r"""PyTorch version of scikit-learn's MinMaxScaler,
    see https://github.com/scikit-learn/scikit-learn/blob/3f89022fa/sklearn/preprocessing/_data.py#L416"""
    def __init__(self,
                 feature_range: Union[Tuple[int], List[int]] = (0,1),
                 clip_transform: bool = False,
                 verbosity: int = 0,
                 **kwargs
                 ):
        self.feature_range = feature_range
        self.scale_ = None
        self.min_ = None
        self.clip_transform = clip_transform
        self.verbosity = verbosity


    def fit(self, x: torch.tensor) -> None:
        if self.verbosity > 0:
            print(f"Fitting MinMaxScaler to data of shape {x.shape}")
        self._reset()
        data_min = torch.min(x, dim=0).values
        data_max = torch.max(x, dim=0).values

        data_range = data_max - data_min
        data_range[data_range == 0.0] = 1.0
        self.scale_ = (self.feature_range[1] - self.feature_range[0]) / data_range
        self.min_ = self.feature_range[0] - data_min * self.scale_


    def fit_transform(self, x: torch.tensor) -> torch.tensor:
        self.fit(x)
        return self.transform(x)


    def transform(self, x: torch.tensor) -> torch.tensor:
        x *= self.scale_.to(x.device)
        x += self.min_.to(x.device)
        if self.clip_transform:
            x.clamp_(self.feature_range[0], self.feature_range[1])
        return x


    def inverse_transform(self, x: torch.tensor) -> torch.tensor:
        if not self._is_fitted():
            raise ValueError("MinMaxScaler has to be fitted first!")
        else:
            x -= self.min_.to(x.device)
            x /= self.scale_.to(x.device)
            return x


    def _is_fitted(self) -> bool:
        if self.scale_ is None or self.min_ is None:
            return False
        else:
            return True


    def _reset(self) -> None:
        self.scale_ = self.min_ = None



class InvertibleMinMaxScaler:
    def __init__(self, mode: str):
        print("Deprecation Warning: InvertibleMinMaxScaler is deprecated, use MinMaxScaler instead!")
        if mode not in ['axis', 'all']:
            raise ValueError(f"Unsupported mode: {mode}. Needs to be either 'axis' or 'all'")
        self.mode = mode
        self.minimums = None
        self.maximums = None


    def __call__(self, t: torch.tensor) -> torch.Tensor:
        self.minimums = torch.min(t, dim=0).values if self.mode == 'axis' else torch.min(t)
        self.maximums = torch.max(t, dim=0).values if self.mode == 'axis' else torch.max(t)
        return (t - self.minimums) / (self.maximums - self.minimums)


    def backwards(self, t: torch.tensor) -> torch.Tensor:
        if self.minimums is None or self.maximums is None:
            raise ValueError("Need to first perform normalization")
        res = t * (self.maximums - self.minimums) + self.minimums
        self.maximums = None
        self.minimums = None
        return res



class ShapeNormalizer:
    def __init__(self, method: str):
        if method not in ['minmax', 'lp']:
            raise ValueError(
                "Supported normalization methods are 'minmax' and 'lp', got "
                f"{method}"
            )
        self.method = method
        self.min_max_scaler = None


    def __call__(self, input: Union[torch.Tensor, np.ndarray], **kwargs) -> Union[torch.Tensor, np.ndarray]:
        inp_type = type(input)
        if isinstance(input, np.ndarray):
            input = torch.from_numpy(input)
        if self.method == 'lp':
            output = torch.nn.functional.normalize(input, **kwargs)
        else:
            self.min_max_scaler = InvertibleMinMaxScaler(**kwargs)
            output = self.min_max_scaler(input)
        if inp_type == np.ndarray:
            output = output.cpu().numpy()
        return output


    def backwards(self, input: Union[torch.Tensor, np.ndarray]) -> Union[torch.Tensor, np.ndarray]:
        if self.method == 'lp':
            raise NotImplementedError('Lp-Normalization backwartds not implemented')
        inp_type = type(input)
        if isinstance(input, np.ndarray):
            input = torch.from_numpy(input)
        output = self.min_max_scaler.backwards(input)
        if inp_type == np.ndarray:
            output = output.cpu().numpy()
        return output



class PoseRepresentationConverter(nn.Module):
    def __init__(self,
                 conversion_method: Callable,
                 pose_only_input: bool,
                 n_shape_params: int = -1
                 ):
        """
            Class that represents a conversion from one pose representation to another.

            Params
            ------
                conversion_method (Callable):
                    The function that will perform the conversion. Should usually either
                    be a nn.Module or nn.Sequential
                pose_only_input (bool):
                    Whether the input to the forward method will contain only pose parameters
                n_shape_params (int):
                    Only required if pose_only_input is False, i.e. the input to the forward
                    method will contain not only pose but also translation and shape parameters.
                    Defaults to -1
        """
        super().__init__()
        self.conv = conversion_method
        self.pose_only = pose_only_input
        self.n_shape_params = n_shape_params


    def forward(self, x):
        if self.pose_only:
            pose = x
        else:
            transl = x[:, :3]
            shape_params = x[:, 3 : (3 + self.n_shape_params)]
            pose = x[:, (3 + self.n_shape_params):]

        conv_output = self.conv(pose)

        if self.pose_only:
            return conv_output
        else:
            return torch.cat((transl, shape_params, conv_output), dim=1)



class NoOp(nn.Module):
    def __init__(self):
        super().__init__()


    def forward(self, x):
        return x



class RotVec2Quat(nn.Module):
    def __init__(self):
        super().__init__()


    def forward(self, x):
        """
            Params
            ------
                x (torch.Tensor):
                    Input rotation vectors of shape (B, n_joints * 3)

            Returns
            -------
                torch.Tensor:
                    Quaternions of shape (B, n_joints * 4)
        """
        # pose rotation vector to quaternion
        quats = self.axis_angle_to_quaternion(x.reshape(x.shape[0], -1, 3))                 # (B, n_joints, 4)
        quats_flattened = quats.reshape(quats.shape[0], quats.shape[1] * quats.shape[2])    # (B, n_joints * 4)
        return quats_flattened


    # BSD license of Pytorch3D
    # Copyright (c) Meta Platforms, Inc. and affiliates. All rights reserved.
    def axis_angle_to_quaternion(self, axis_angle: torch.Tensor) -> torch.Tensor:
        """
            Convert rotations given as axis/angle to quaternions.

            Params
            ------
                axis_angle (torch.Tensor):
                    Rotations given as a vector in axis angle form,
                    as a tensor of shape (..., 3), where the magnitude is
                    the angle turned anticlockwise in radians around the
                    vector's direction.

            Returns
            -------
                torch.Tensor:
                    quaternions with real part first, as tensor of shape (..., 4).
        """
        angles = torch.norm(axis_angle, p=2, dim=-1, keepdim=True)
        half_angles = angles * 0.5
        eps = 1e-6
        small_angles = angles.abs() < eps
        sin_half_angles_over_angles = torch.empty_like(angles)
        sin_half_angles_over_angles[~small_angles] = (
            torch.sin(half_angles[~small_angles]) / angles[~small_angles]
        )
        # for x small, sin(x/2) is about x/2 - (x/2)^3/6
        # so sin(x/2)/x is about 1/2 - (x*x)/48
        sin_half_angles_over_angles[small_angles] = (
            0.5 - (angles[small_angles] * angles[small_angles]) / 48
        )
        quaternions = torch.cat(
            [torch.cos(half_angles), axis_angle * sin_half_angles_over_angles], dim=-1
        )
        return quaternions



class Quat2RotVec(nn.Module):
    def __init__(self):
        super().__init__()


    def forward(self, x):
        # pose quaternions to rotation vector
        rot_vecs = self.quaternion_to_axis_angle(x.reshape(x.shape[0], -1, 4))                  # (B, n_joints, 3)
        rot_vecs = rot_vecs.reshape(rot_vecs.shape[0], rot_vecs.shape[1] * rot_vecs.shape[2])   # (B, n_joints * 3)
        return rot_vecs


    # BSD license of Pytorch3D
    # Copyright (c) Meta Platforms, Inc. and affiliates. All rights reserved.
    def quaternion_to_axis_angle(self, quaternions: torch.Tensor) -> torch.Tensor:
        """
            Convert rotations given as quaternions to axis/angle.

            Params
            ------
                quaternions (torch.Tensor):
                    quaternions with real part first, as tensor of shape (..., 4).

            Returns
            -------
                torch.Tensor:
                    Rotations given as a vector in axis angle form, as a tensor
                    of shape (..., 3), where the magnitude is the angle
                    turned anticlockwise in radians around the vector's
                    direction.
        """
        norms = torch.norm(quaternions[..., 1:], p=2, dim=-1, keepdim=True)
        half_angles = torch.atan2(norms, quaternions[..., :1])
        angles = 2 * half_angles
        eps = 1e-6
        small_angles = angles.abs() < eps
        sin_half_angles_over_angles = torch.empty_like(angles)
        sin_half_angles_over_angles[~small_angles] = (
            torch.sin(half_angles[~small_angles]) / angles[~small_angles]
        )
        # for x small, sin(x/2) is about x/2 - (x/2)^3/6
        # so sin(x/2)/x is about 1/2 - (x*x)/48
        sin_half_angles_over_angles[small_angles] = (
            0.5 - (angles[small_angles] * angles[small_angles]) / 48
        )
        return quaternions[..., 1:] / sin_half_angles_over_angles



class RotMatFrom6D(nn.Module):
    def __init__(self,
                 use_gram_schmidt: bool = False):
        """
            Calculates a 3x3 rotation matrix from its 6D representation using
            either Gram-Schmidt or Cross-product method.

            Params
            ------
                use_gram_schmidt (bool):
                    Use Gram-Schmidt orthonormalization scheme (slower than
                    alternative cross-product method). Optional, defaults to False
        """
        super().__init__()
        self.use_gram_schmidt = use_gram_schmidt


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
            Calculates the 3x3 rotation matrix representation of the given
            6D representation.

            Params
            ------
                x (torch.Tensor):
                    6D rotation representation in the form of
                    (batch_size, 6 * number_joints) where
                    the 6D representation is given in row-major order

            Returns
            -------
                torch.Tensor:
                    The input tensor with 3x3 rotation matrix representation of the
                    given 6D rotation representation with shape
                    (batch_size, number_joints * 9) where
                    the rotation matrix is stored in row-major order.

        """
        if x.ndim == 1:
            x = x.unsqueeze(0)

        bs = x.shape[0]
        n_joints = x.shape[1] // 6
        x = x.reshape(bs, n_joints, 6)      # (B, J, 6)

        col_vec_1 = x[..., 0::2]      # (B, J, 3)
        col_vec_2 = x[..., 1::2]      # (B, J, 3)

        if self.use_gram_schmidt:
            a, b, c = self._gram_schmidt_method(col_vec_1, col_vec_2)
        else:
            a, b, c = self._cross_product_method(col_vec_1, col_vec_2)
        # a, b, and c are each of shape (B, J, 3)
        rot_mat = torch.stack((a, b, c), dim=-1) # (B, J, 3, 3) with first column vector a, then b, then c
        res_pose = rot_mat.reshape(bs, -1)  # (B, 9 * n_joints)
        return res_pose


    def _projection(self,
                    u: torch.Tensor,
                    v: torch.Tensor
                    ) -> torch.Tensor:
        """Projects v onto u"""
        # v and u have shape (B, J, 3)
        return (v*u).sum(dim=-1).unsqueeze(-1) / (u*u).sum(dim=-1).unsqueeze(-1) * u


    def _are_collinear(batch_1, batch_2, tol=1e-6):
        """
            Checks whether any two vectors from the given batches are collinear

            Params
            ------
                batch_1 (torch.Tensor):
                    First batch of vectors of shape (B, J, 3)
                batch_2 (torch.Tensor):
                    Second batch of vectors of shape (B, J, 3)

            Returns
            -------
                torch.Tensor:
                    A mask of shape (B, J) that contains boolean elements indicating
                    whether the two vectors at this position of collinear (True)
                    or not (False)
        """
        cross_norm = torch.cross(batch_1, batch_2, dim=-1).norm(dim=-1)
        collinear_mask = cross_norm < tol
        return collinear_mask


    def _get_linearly_independent_vectors(self, a_1, a_2, tol=1e-6):
        """
            Based on the given two batches of vectors, returns a batch that is guaranteed
            to be linearly independent of the first batch a_1. If a_1 and a_2 are
            linearly independent, a_2 would be returned

            Params
            ------
                a_1 (torch.Tensor):
                    First batch of vectors of shape (B, J, 3)
                a_2 (torch.Tensor):
                    Second batch of vectors of shape (B, J, 3)

            Returns
            -------
                torch.Tensor:
                    A tensor of shape (B, J, 3) that contains linearly independent
                    vectors from a_1.
        """
        collinear_mask_input = self._are_collinear(a_1, a_2, tol)

        candidate_1 = torch.tensor([1.0, 0.0, 0.0], device=a_1.device, dtype=a_1.dtype).view(1, 1, 3)
        candidate_2 = torch.tensor([0.0, 1.0, 0.0], device=a_1.device, dtype=a_1.dtype).view(1, 1, 3)
        B, J, _ = a_1.shape
        candidate_1.expand(B, J, 3)
        candidate_2.expand(B, J, 3)
        collinear_mask_candidate_1 = self._are_collinear(a_1, candidate_1, tol=tol)
        replacement = torch.where(collinear_mask_candidate_1.unsqueeze(-1), candidate_2, candidate_1)

        new_a_2 = torch.where(collinear_mask_input.unsqueeze(-1), replacement, a_2)
        return new_a_2


    def _cross_product_method(self, a_1, a_2):
        """
            Uses the cross product method to calculate a valid orthonormal matrix from given
            vectors a_1 and a_2.

            Params
            ------
                a_1 (torch.Tensor):
                    First batch of column vectors of shape (B, J, 3)
                a_2 (torch.Tensor):
                    Second batch of column vectors of shape (B, J, 3)

            Returns
            -------
                torch.Tensor:
                    First column vector of the orthonormal 3x3 matrix
                torch.Tensor:
                    Second column vector of the orthonormal 3x3 matrix
                torch.Tensor:
                    Third column vector of the orthonormal 3x3 matrix
        """
        # _get_linearly_independent_vectors is not differentiable, so ignore this edge case
        #a_2 = self._get_linearly_independent_vectors(a_1, a_2)

        u_1 = torch.nn.functional.normalize(a_1, dim=-1)
        u_3 = torch.nn.functional.normalize(
            torch.linalg.cross(u_1, a_2),
            dim=-1
        )
        u_2 = torch.linalg.cross(u_3, u_1)
        return u_1, u_2, u_3


    def _gram_schmidt_method(self, a_1, a_2):
        u_1 = torch.nn.functional.normalize(a_1, dim=-1)
        u_2 = torch.nn.functional.normalize(
            a_2 - self._projection(u_1, a_2),
            dim=-1
        )
        u_3 = torch.linalg.cross(u_1, u_2)
        return u_1, u_2, u_3



class RotMat2RotVec(nn.Module):
    def __init__(self):
        """
            Converts a 3x3 rotation matrix to its rotation vector representation
        """
        super().__init__()


    def forward(self, x: torch.Tensor):
        """
            Converts the given rotation matrices to rotation vectors

            Params
            ------
                x (torch.Tensor):
                    Rotation matrices of shape (B, n_joints * 9) which should be
                    converted to rotation vectors. The rotation matrices are
                    expected in column-major order.

            Returns
            -------
                torch.Tensor:
                    Input with rotation matrices converted to rotation vectors.
                    Shape (B, 3 * n_joints)
        """
        if x.ndim == 1:
            x = x.unsqueeze(0)
        bs = x.shape[0]

        rot_mats = x.reshape(-1, 3, 3)
        #rot_vecs = batch_rot2aa(rot_mats)   # (B * n_joints, 3)
        #print(f"Shape: {rot_vecs.shape}", flush=True)
        quats = batched_rot_mat_to_quat(rot_mats)
        conv = Quat2RotVec()
        rot_vecs = conv.quaternion_to_axis_angle(quats)

        rot_vecs = rot_vecs.reshape(bs, -1)  # (B, 3 * n_joints)
        return rot_vecs



class RotVec2RotMat(nn.Module):
    def __init__(self):
        """
            Converts the given rotation vectors to their 3x3 rotation matrix
            representation.
        """
        super().__init__()


    def forward(self, x: torch.Tensor):
        """
            Converts the given rotation vectors to their 3x3 rotation matrix
            representations.

            Params
            ------
                x (torch.Tensor):
                    Rotation vectors of shape (B, 3 * n_joints)

            Returns
            -------
                torch.Tensor:
                    The input with pose parameters converted to row-major
                    3x3 rotation matrices of shape (B, 9 * n_joints)
        """
        if x.ndim == 1:
            x = x.unsqueeze(0)
        bs = x.shape[0]

        rot_vecs = x.reshape(-1, 3)                             # (B * n_joints, 3)
        rot_mats = batch_rodrigues(rot_vecs)                    # (B * n_joints, 3, 3)
        rot_mats = rot_mats.reshape(bs, -1)   # (B, 9 * n_joints)
        return rot_mats



class RotMatTo6D(nn.Module):
    def __init__(self):
        """
            Converts the given 3x3 rotation matrices to their continuous
            6D rotation representations.
        """
        super().__init__()


    def forward(self, x: torch.Tensor):
        """
            Converts the given rotation matrices to their continuous 6D
            rotation representations.

            Params
            ------
                x (torch.Tensor):
                    3x3 rotation matrices of shape (B, 9 * n_joints) with
                    rotation matrices in row-major order

            Returns
            -------
                torch.Tensor:
                    The input with 3x3 rotation matrices converted to their
                    continuous 6D representations (row-major order) of shape
                    (B, 6 * n_joints)
        """
        if x.ndim == 1:
            x.unsqueeze_(0)
        bs = x.shape[0]

        x = x.reshape(bs, -1, 3, 3) # (B, n_joints, 3, 3)
        # Get first two column vectors only
        cont_repr = x[..., :2].reshape(bs, -1)  # (B, 6 * n_joints)
        return cont_repr


class RotMatToQuat(nn.Module):
    def __init__(self):
        """
            Converts the given 3x3 rotation matrices to their quaternion
            representation.
        """
        super().__init__()


    def forward(self, x: torch.Tensor):
        """
            Converts the given rotation matrices to quaternion representation.

            Params
            ------
                x (torch.Tensor):
                    3x3 rotation matrices of shape (B, 9 * n_joints) with
                    rotation matrices in column-major order
        """
        raise NotImplementedError("Rotation matrix to quaternion convertor not implemented")
        pass


NORMALIATION_METHODS = DotMap({
    'minmax': MinMaxScaler
}, _dynamic=False)

def get_data_transform_class(input_rotation_representation: PoseRepresentation,
                             output_rotation_representation: PoseRepresentation,
                             num_shape_components: int
                             ) -> Tuple[Union[None, NoOp, RotVec2Quat]]:
    """
        Returns the transformation classes to convert data from the dataset
        (assumed to be in rotation vector representation) to given input rotation
        representation and network output (assumed to be in given output rotation
        representation) to rotation vector representation.

        Params
        ------
            input_rotation_representation (PoseRepresentation):
                Rotation representation that the network should use as input
            output_rotation_representation (PoseRepresentation):
                Rotation representation that the network output is encoded as
            num_shape_components (int):
                Number of shape components.

        Returns
        -------
            nn.Module:
                Transformation from dataset's rotation vector representation to
                required network input representation
            nn.Module:
                Transformation from network output representation to dataset's rotation
                vector representation

        Raises
        ------
            NotImplementedError:
                If the given pose representation is not supported
    """
    # Assumes default data is in rotation vector format
    inp = outp = None
    if input_rotation_representation == PoseRepresentation.ROTATION_VECTOR:
        inp = NoOp()
    elif input_rotation_representation == PoseRepresentation.QUATERNION:
        inp = RotVec2Quat()
    elif input_rotation_representation == PoseRepresentation.ROTATION_MATRIX_9D:
        inp = RotVec2RotMat()
    elif input_rotation_representation == PoseRepresentation.ROTATION_MATRIX_6D:
        inp = nn.Sequential(RotVec2RotMat(),
                            RotMatTo6D())
    else:
        raise NotImplementedError(f"The given input rotation representation is currently not supported: {input_rotation_representation}")

    if output_rotation_representation == PoseRepresentation.ROTATION_VECTOR:
        outp = NoOp()
    elif output_rotation_representation == PoseRepresentation.QUATERNION:
        outp = Quat2RotVec()
    elif output_rotation_representation == PoseRepresentation.ROTATION_MATRIX_9D:
        outp = RotMat2RotVec()
    elif output_rotation_representation == PoseRepresentation.ROTATION_MATRIX_6D:
        outp = nn.Sequential(RotMatFrom6D(),
                             RotMat2RotVec())
    else:
        raise NotImplementedError(f"The given output rotation representation is currently not supported: {output_rotation_representation}")
    input_conv = PoseRepresentationConverter(inp, False, num_shape_components)
    output_conv = PoseRepresentationConverter(outp, False, num_shape_components)
    return input_conv, output_conv



def get_rotation_representation_conversion(input_rot_repr: PoseRepresentation,
                                           target_rot_repr: PoseRepresentation
                                           ) -> PoseRepresentationConverter:
    """
        Returns a callable that can be used to convert from one rotation representation to another.
        This function expects that to-be-converted input only consists of pose parameters

        Params
        ------
            input_rot_repr (PoseRepresentation):
                Origin rotation representation
            target_rot_repr (PoseRepresentation):
                Target rotation representation

        Returns
        -------
            PoseRepresentationConverter:
                nn.Module that can be used to convert from one rotation representation to another

        Raises
        ------
            NotImplementedError:
                If the requested conversion is not implemented
    """
    func = None

    if input_rot_repr == target_rot_repr:
        func = NoOp()

    elif input_rot_repr == PoseRepresentation.ROTATION_VECTOR:
        if target_rot_repr == PoseRepresentation.QUATERNION:
            func = RotVec2Quat()
        elif target_rot_repr == PoseRepresentation.ROTATION_MATRIX_9D:
            func = RotVec2RotMat()
        elif target_rot_repr == PoseRepresentation.ROTATION_MATRIX_6D:
            func = nn.Sequential(RotVec2RotMat(), RotMatTo6D())

    elif input_rot_repr == PoseRepresentation.QUATERNION:
        if target_rot_repr == PoseRepresentation.ROTATION_VECTOR:
            func = Quat2RotVec()
        elif target_rot_repr == PoseRepresentation.ROTATION_MATRIX_9D:
            func = nn.Sequential(Quat2RotVec(), RotVec2RotMat())
        elif target_rot_repr == PoseRepresentation.ROTATION_MATRIX_6D:
            func = nn.Sequential(Quat2RotVec(), RotVec2RotMat(), RotMatTo6D())

    elif input_rot_repr == PoseRepresentation.ROTATION_MATRIX_9D:
        if target_rot_repr == PoseRepresentation.ROTATION_VECTOR:
            func = RotMat2RotVec()
        elif target_rot_repr == PoseRepresentation.QUATERNION:
            func = nn.Sequential(RotMat2RotVec(), RotVec2Quat())
        elif target_rot_repr == PoseRepresentation.ROTATION_MATRIX_6D:
            func = RotMatTo6D()

    elif input_rot_repr == PoseRepresentation.ROTATION_MATRIX_6D:
        if target_rot_repr == PoseRepresentation.ROTATION_VECTOR:
            func = nn.Sequential(RotMatFrom6D(), RotMat2RotVec())
        elif target_rot_repr == PoseRepresentation.QUATERNION:
            func = nn.Sequential(RotMatFrom6D(), RotMat2RotVec(), RotVec2Quat())
        elif target_rot_repr == PoseRepresentation.ROTATION_MATRIX_9D:
            func = RotMatFrom6D()

    if func is None:
        raise NotImplementedError(f"Conversion from {input_rot_repr} to {target_rot_repr} is not implemented!")
    conv = PoseRepresentationConverter(func, True)
    return conv



class RotationVectorCanonizer(nn.Module):
    def __init__(self):
        """
            Canonizes a rotation vector such that the rotation angle is always
            in the interval [0, pi].
        """
        super().__init__()


    def count_non_canonicals(self, vecs: torch.Tensor) -> int:
        """
            Returns the number of rotation vectors in the given batch that have
            an angle greater than pi

            Params
            ------
                vecs (torch.Tensor):
                    Batch of rotation vectors of shape (B, n_joints * 3)

            Returns
            -------
                int:
                    Number of rotation vectors in the batch who's angle is greater than pi.
        """
        vecs = vecs.reshape(-1, 3)
        angles = torch.linalg.norm(vecs, dim=-1, keepdim=False)
        greater_pi = angles > torch.pi
        return torch.sum(greater_pi, dim=-1).item()


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
            Canonizes the given rotation vectors.

            Params
            ------
                x (torch.Tensor):
                    Rotation vectors of shape (B, n_joints * 3)

            Returns
            -------
                torch.Tensor:
                    Canonized rotation vectors of shape (B, n_joints * 3)
        """
        bs = x.shape[0]
        x = x.reshape(-1, 3)
        normalized = torch.nn.functional.normalize(x, p=2, dim=-1)
        angles = torch.linalg.norm(x, dim=-1, keepdim=True)
        # Restrict angles to [0, 2pi]
        angles = torch.remainder(angles, 2*torch.pi)
        # Convert rotation vectors with angles > pi: Flip axis and change angle
        greater_pi_mask = angles > torch.pi
        angles = torch.where(greater_pi_mask, 2*torch.pi - angles, angles)
        normalized = torch.where(greater_pi_mask, normalized * -1, normalized)
        return (normalized * angles).reshape(bs, -1)



def canonize_rotation_vectors_numpy(rot_vecs: np.ndarray):
    """
        Calculates canonical rotation vectors (angles in [0, pi]) from the
        given SMPL rotation vectors

        Params
        ------
            rot_vecs (numpy.ndarray):
                SMPL pose parameters in rotation vector form of shape (B, n_joints * 3)

        Returns
        -------
            The input SMPL parameters in canonical rotation vector from.
            Shape (B, n_joints * 3)
    """
    bs = rot_vecs.shape[0]
    rot_vecs = rot_vecs.reshape(-1, 3)
    angles = np.linalg.norm(rot_vecs, axis=-1, keepdims=True)
    angles_div = np.copy(angles)
    angles_div[np.isclose(angles_div, np.zeros_like(angles_div))] = 1
    normalized = rot_vecs / angles_div

    # Restrict angles to [0, 2pi]
    angles = np.remainder(angles, 2*np.pi)
    # Convert rotation vectors with angles > pi to [0, pi]
    greater_pi_mask = angles > np.pi
    angles = np.where(greater_pi_mask, 2*np.pi - angles, angles)
    normalized = np.where(greater_pi_mask, normalized * -1, normalized)
    return (normalized * angles).reshape(bs, -1)


def count_non_canonical_rotation_vectors_numpy(rot_vecs: np.ndarray) -> int:
    """
        Returns the number of non-canonical rotation vectors (i.e. angle outside of [0, pi])
        for the given batch of rotation vectors

        Params
        ------
            rot_vecs (numpy.ndarray):
                Batch of rotation vectors of shape (N, n_joints * 3)

        Returns
        -------
            int: The number of non-canonical rotation vectors in the input
    """
    rot_vecs = rot_vecs.reshape(-1, 3)
    angles = np.linalg.norm(rot_vecs, axis=-1)
    greater_pi = angles > np.pi
    return np.sum(greater_pi, axis=-1)
