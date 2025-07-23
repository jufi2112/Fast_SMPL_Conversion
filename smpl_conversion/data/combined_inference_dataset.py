# Copyright (c) 2025 Julien Fischer, Chair of Computer Graphics and Visualization, TUD Dresden University of Technology.
# All rights reserved.
# Licensed under the MIT License. See LICENSE in the project root for license information.

import torch
import numpy as np

from os import path as osp
from torch.utils.data import Dataset
from typing import Union, List, Dict, Tuple, Any
from smpl_conversion.utils.enum_configurations import PoseRepresentation
from smpl_conversion.data.transforms import get_rotation_representation_conversion, PoseRepresentationConverter

class CombinedInferenceDataset(Dataset):
    """
        Dataset for combined inference. At the moment, does not support
        any data normalization.

        Params
        ------
            dataset_fpath (str):
                Path to the dataset. If the input is a .npy file, it's expected to
                contain all parameters horizontally stacked like this:
                [translation, shape, pose]. Otherwise, provide the translation_key,
                shape_key, and pose_key parameters
            device_loading (str):
                Device onto which the whole dataset is loaded when reading it.
            device_detch (str):
                Device onto which the requested items (__getitem__) should be
                transferred.
            n_shape_components (int):
                Number of shape components
            rotation_representation (PoseRepresentation):
                Rotation representation the pose parameters should have.
            translation_key (str):
                Key under which the translation parameters are saved
            shape_key (str):
                Key under which the shape parameters are saved
            pose_key (str):
                Key under which the pose parameters are saved
            subsampling_factor (int):
                Factor by which the dataset should be subsampled. Defaults to 1
                (i.e. no subsampling).
    """
    def __init__(self,
                 dataset_fpath: str,
                 device_loading: str,
                 device_fetch: str,
                 n_shape_components: int,
                 rotation_representation: PoseRepresentation,
                 translation_key: str = None,
                 shape_key: str = None,
                 pose_key: str = None,
                 subsampling_factor: int = 1
                 ):
        self.device_loading = device_loading
        self.device_fetch = device_fetch
        self.rot_repr = rotation_representation
        self.rot_conv: PoseRepresentationConverter = get_rotation_representation_conversion(PoseRepresentation.ROTATION_VECTOR,
                                                                                            self.rot_repr).to(self.device_loading)
        self.data = self._load_dataset(dataset_fpath,
                                       translation_key,
                                       shape_key,
                                       pose_key,
                                       subsampling_factor,
                                       n_shape_components)


    def __len__(self) -> int:
        return self.data.shape[0]


    def __getitem__(self, idx) -> torch.Tensor:
        return self.data[idx].to(self.device_fetch)


    def _load_dataset(self,
                      file_path: str,
                      key_trans: str,
                      key_shape: str,
                      key_pose: str,
                      subsampling_factor: int,
                      n_shape_components: int
                      ) -> torch.Tensor:
        """
            Loads the dataset by looking for the specified keys and stacking
            them horizontally [translation, shape, pose]. If the provided
            dataset is already .npy file, expects the data to already be
            stacked like this.

            Params
            ------
                file_path (str):
                    Path to the dataset
                key_trans (str):
                    Translation key
                key_shape (str):
                    Shape key
                key_pose (str):
                    Pose key
                subsampling_factor (int):
                    Subsampling factor

            Returns
            -------
                torch.Tensor:
                    The loaded data as stacked tensor: [trans, betas, poses]
                    of shape (N, 3 + #shapes + #poses)

            Raises
            ------
                ValueError:
                    If the provided keys do not exist in the dataset.
        """
        if osp.splitext(file_path)[1] == '.npy':
            data = np.load(file_path)
            poses = data[:, 3+n_shape_components:]
            poses = self.rot_conv(poses)
            data = torch.tensor(np.hstack((data[:, :3+n_shape_components], poses))).to(dtype=torch.float32, device=self.device_loading)
        else:
            with np.load(file_path) as file:
                file_keys = list(file.keys())
                if key_trans not in file_keys:
                    raise ValueError(f"The translation key {key_trans} could not be found in the provided dataset!")
                if key_shape not in file_keys:
                    raise ValueError(f"The shape key {key_shape} could not be found in the provided dataset!")
                if key_pose not in file_keys:
                    raise ValueError(f"The pose key {key_pose} could not be found in the provided dataset!")
                poses = torch.tensor(file[key_pose][::subsampling_factor, :], dtype=torch.float32)
                poses_conv = self.rot_conv(poses)
                del poses
                data = torch.hstack(
                    (
                        torch.tensor(file[key_trans][::subsampling_factor, :], dtype=torch.float32),
                        torch.tensor(file[key_shape][::subsampling_factor, :n_shape_components], dtype=torch.float32),
                        poses_conv
                    )).to(dtype=torch.float32, device=self.device_loading)
        return data
