# Copyright (c) 2025 Julien Fischer, Chair of Computer Graphics and Visualization, TUD Dresden University of Technology.
# All rights reserved.
# Licensed under the MIT License. See LICENSE in the project root for license information.

import torch
import numpy as np
from dotmap import DotMap
from typing import Optional
from smpl_conversion.utils.Quaternions.quat import Quat
from smpl_conversion.utils.model_infos import MODEL_STATS
from smpl_conversion.utils.enum_configurations import BodyModelType

def sample_random_translation(n_samples: int,
                              uniform: DotMap = None,
                              normal: DotMap = None
                              ) -> torch.Tensor:
    """
        Randomly samples translations of shape (n_samples, 3) from either
        normal or uniform distributions with the given parameters.

        Params
        ------
            n_samples (int):
                Number of translations to sample, i.e. batch size
            uniform (DotMap):
                If provided, the parameters of the uniform distribution from which
                samples should be drawn. Defaults to None
            normal (DotMap):
                If provided, the parameters of the normal distribution from which
                samples should be drawn. Defaults to None

        Returns
        -------
            torch.Tensor:
                Tensor of shape (n_samples, 3) with elements drawn from the specified
                probability distribution
    """
    if (uniform is None and normal is None) or (uniform is not None and normal is not None):
        raise ValueError("Must provide parameters for either uniform or normal distribution")
    if normal is not None:
        mean_x = torch.ones((n_samples, 1), dtype=torch.float32) * torch.tensor(normal.x_mean, dtype=torch.float32)
        std_x = torch.ones((n_samples, 1), dtype=torch.float32) * torch.tensor(normal.x_std, dtype=torch.float32)
        mean_y = torch.ones((n_samples, 1), dtype=torch.float32) * torch.tensor(normal.y_mean, dtype=torch.float32)
        std_y = torch.ones((n_samples, 1), dtype=torch.float32) * torch.tensor(normal.y_std, dtype=torch.float32)
        mean_z = torch.ones((n_samples, 1), dtype=torch.float32) * torch.tensor(normal.z_mean, dtype=torch.float32)
        std_z = torch.ones((n_samples, 1), dtype=torch.float32) * torch.tensor(normal.z_std, dtype=torch.float32)
        return torch.cat(
            [
                torch.normal(mean_x, std_x),
                torch.normal(mean_y, std_y),
                torch.normal(mean_z, std_z)
            ],
            dim=1
        )
    if uniform is not None:
        return torch.cat(
            [
                torch.FloatTensor(n_samples, 1).uniform_(uniform.x_min, uniform.x_max),
                torch.FloatTensor(n_samples, 1).uniform_(uniform.y_min, uniform.y_max),
                torch.FloatTensor(n_samples, 1).uniform_(uniform.z_min, uniform.z_max)
            ],
            dim=1
        )



def sample_random_shape(n_samples: int,
                        n_shape_components: int,
                        uniform: DotMap = None,
                        normal: DotMap = None
                        ) -> torch.Tensor:
    """
        Randomly samples shape components of shape (n_samples, n_shape_components)
        from either uniform or normal distribution with the given parameters.

        Params
        ------
            n_samples (int):
                Number of samples to draw, i.e. batch size.
            n_shape_components (int):
                Number of shape components to draw for each sample.
            uniform (DotMap):
                If provided, the parameters of the uniform distribution from
                which samples should be drawn. Defaults to None.
            normal (DotMap):
                If provided, the parameters of the normal distribution from
                which samples should be drawn. Defaults to None.

        Returns
        -------
            torch.Tensor:
                Tensor of shape (n_samples, n_shape_components) with elements drawn
                from the specified probability distribution.
    """
    if (uniform is None and normal is None) or (uniform is not None and normal is not None):
        raise ValueError("Must provide parameters for either uniform or normal distribution")
    if normal is not None:
        means = torch.ones((n_samples, n_shape_components), dtype=torch.float32) * torch.tensor(normal.mean, dtype=torch.float32)
        stds = torch.ones((n_samples, n_shape_components), dtype=torch.float32) * torch.tensor(normal.std, dtype=torch.float32)
        return torch.normal(means, stds)
    elif uniform is not None:
        return torch.FloatTensor(n_samples, n_shape_components).uniform_(uniform["from"], uniform.to)
    else:
        return None



def sample_random_pose(n_samples: int,
                       body_type: BodyModelType,
                       np_rng: np.random.Generator,
                       n_joints: Optional[int] = None
                       ) -> torch.Tensor:
    """
        Randomly samples uniformly distributed rotation vectors of shape (n_samples, x) where
        x depends on the number of joints in the input body model.
        Uniform sampling is done via quaternions.

        Params
        ------
            n_samples (int):
                Number of samples to draw, i.e. batch size.
            body_type (BodyModelType):
                Type of body model for which random pose should be sampled
            np_rng (np.random.Generator):
                Numpy random generator instance that should be used
            n_joints (int):
                If provided, samples random rotations for the given
                number of joints. If not provided, automatically
                uses the input body model type's number of joints.

        Returns
        -------
            torch.Tensor:
                Tensor of shape (n_samples, #body_model_joints * 3)
    """
    if n_joints is None:
        n_joints = MODEL_STATS[body_type].joints
    q = Quat.generate_random_rotations(num_samples=n_samples * n_joints,
                                       return_float_array=True,
                                       np_rng=np_rng
                                       )
    return torch.tensor(Quat.to_rot_vec(q).reshape(n_samples, n_joints*3), dtype=torch.float32)
