# Copyright (c) 2025 Julien Fischer, Chair of Computer Graphics and Visualization, TUD Dresden University of Technology.
# All rights reserved.
# Licensed under the MIT License. See LICENSE in the project root for license information.

import torch
import numpy as np

from dotmap import DotMap
from torch.utils.data import Dataset
from typing import Union, List, Dict, Tuple, Any
from smpl_conversion.data.transforms import (
    NORMALIATION_METHODS,
    PoseRepresentationConverter,
    get_rotation_representation_conversion
)
from smpl_conversion.utils.enum_configurations import BodyModelType, PoseRepresentation

class SMPLConversionDataset(Dataset):
    def __init__(
            self,
            dataset_fpath: str,
            mode: str,
            parameters: List[str],
            parameters_to_learn: List[str],
            gender: str,
            device_loading: str,
            device_fetch: str,
            rotation_representation: PoseRepresentation,
            ground_truth_rotation_representation: PoseRepresentation = None,
            subsampling_factor: int = 1,
            transform = None,
            target_transform = None,
            normalize_data: bool = False,
            normalization_method: str = 'minmax', # 'minmax'
            normalization_parameters: Dict = None,
            allow_unsupervised: bool = True
            ):
        """
            Dataset for parameter conversion of models from the SMPL family and SUPR.

            Params
            ------
                dataset_fpath (str):
                    Dataset file where parameters should be loaded from.
                mode (str):
                    Conversion mode, e.g. "smplx2smplh"
                parameters (list of str):
                    List of parameters that should be extracted from the provided file and contained in the dataset.
                parameters_to_learn (list of str):
                    Parameters that will be normalized. All parameters in argument 'parameters' that are
                    not in 'parameters_to_learn' will not be normalized, even if 'normalize_data=True'.
                gender (str):
                    Gender to extract from the dataset file.
                device_loading (str):
                    Device onto which to load the whole dataset. If the dataset is large (> GPU memory), you can set
                    this to "cpu" and use 'device_fetch' to still load samples onto the GPU later on.
                device_fetch (str):
                    Device onto which items requested (__getitem__) from the dataset should be transferred.
                rotation_representation (PoseRepresentation):
                    Pose rotation representation that data provided by this dataset should be represented in.
                    Will internally perform conversion from rotation vectors (assumed to be the default data representation)
                    to this representation.
                ground_truth_rotation_representation (PoseRepresentation):
                    Pose representation for potential ground truth data. Defaults to None (i.e. use same as
                    rotation_representation argument).
                subsampling_factor (int):
                    Slicing step when selecting data from dataset array. Optional, defaults to 1
                transform (Callable):
                    Not Implemented, transform to apply to all source items.
                target_transform (Callable):
                    Not Implemented, transform to apply to all target items.
                normalize_data (bool):
                    Whether parameters specified in 'parameters_to_learn' should be normalized. Defaults to False.
                normalization_method (str):
                    Normalization method to use if data should be normalized. Defaults to 'minmax'.
                normalization_parameters (dict):
                    Kwargs to the constructor of the normalization method provided above. Defaults to None.
                allow_unsupervised (bool):
                    Optional. When True, allows dataset to not contain entries for the target body model type.
                    Defaults to True.
        """
        self.mode_from, self.mode_to = mode.split("2")
        self.mode_from = BodyModelType.from_string(self.mode_from)
        self.mode_to = BodyModelType.from_string(self.mode_to)
        if gender.lower() not in ['male', 'female', 'neutral']:
            raise ValueError(
                "Supported genders are 'male', 'female', and 'neutral'"
            )
        self.is_unsupervised = False
        self.allow_unsupervised = allow_unsupervised
        self.gender = gender.lower()
        self.device_loading = device_loading
        self.device_fetch = device_fetch
        self.parameters = parameters
        self.parameters_to_learn = parameters_to_learn if isinstance(parameters_to_learn, list) else [parameters_to_learn]
        self.subsampling = subsampling_factor
        self.normalization_method = normalization_method
        self.rot_repr_conv: PoseRepresentationConverter = get_rotation_representation_conversion(PoseRepresentation.ROTATION_VECTOR,
                                                                                                 rotation_representation).to(self.device_loading)
        if ground_truth_rotation_representation is None:
            self.gt_rot_repr_conv = self.rot_repr_conv
        else:
            self.gt_rot_repr_conv: PoseRepresentationConverter = get_rotation_representation_conversion(PoseRepresentation.ROTATION_VECTOR,
                                                                                                        ground_truth_rotation_representation).to(self.device_loading)
        self.data, self.keys_from, self.keys_to = self._load_dataset(dataset_fpath)
        # Setup normalization
        if normalize_data:
            if normalization_method not in NORMALIATION_METHODS.keys():
                raise ValueError(f"Unsupported normalization method: {normalization_method}, expected one of {list(NORMALIATION_METHODS.keys())}!")
            if normalization_parameters is None:
                normalization_parameters = {}
            self.normalizers = self._create_normalizers(normalization_method, **normalization_parameters)
        else:
            self.normalizers = None
        if transform is not None or target_transform is not None:
            raise NotImplementedError('Source and target transform currently not implemented')
        #self.transform = transform
        #self.target_transform = target_transform


    def __len__(self) -> int:
        return len(self.data[self.keys_from[0]])


    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor]:
        """If enabled, normalization is applied to returned values which should be learned"""
        X = DotMap(_dynamic=False)
        y = DotMap(_dynamic=False)
        for key_from in self.keys_from:
            param_name = key_from.split('_')[1]
            if param_name == 'thetas':
                param_name = 'poses'
            if param_name == 'poses':
                X[param_name] = self.rot_repr_conv(self.data[key_from][idx].to(self.device_fetch)).flatten()
            else:
                X[param_name] = self.data[key_from][idx].to(self.device_fetch)
            if self.normalizers is not None and key_from in self.normalizers.keys():
                X[param_name] = self.normalizers[key_from].transform(X[param_name])
        for key_to in self.keys_to:
            if key_to not in self.data.keys() and self.allow_unsupervised:
                continue
            param_name = key_to.split('_')[1]
            if param_name == 'thetas':
                param_name = 'poses'
            if param_name == 'poses':
                y[param_name] = self.gt_rot_repr_conv(self.data[key_to][idx].to(self.device_fetch))
            else:
                y[param_name] = self.data[key_to][idx].to(self.device_fetch)

        # if self.transform is not None:
        #     X = self.transform(X)
        # if self.target_transform is not None:
        #     y = self.target_transform(y)
        return X, y


    def _load_dataset(self, file_path: str) -> DotMap[str, torch.Tensor]:
        data = DotMap()
        keys_from = [
            f"{self.mode_from.to_internal_string()}_{param}_{self.gender}"
            for param in self.parameters
        ]
        keys_to = [
            f"{self.mode_to.to_internal_string()}_{param}_{self.gender}"
            for param in self.parameters
        ]
        keys_not_found = []
        with np.load(file_path) as file:
            for key_from in keys_from:
                if key_from not in list(file.keys()):
                    keys_not_found.append(key_from)
                else:
                    params = torch.tensor(file[key_from][::self.subsampling],
                                          dtype=torch.float32,
                                          device=self.device_loading)
                    if key_from.split('_')[1] == 'poses':
                        pass # params = self.rot_repr_conv(params) # Do this in __getitem__ function for each batch
                    if key_from in data:
                        data[key_from] = torch.vstack((data[key_from], params))
                    else:
                        data[key_from] = params
                # Check whether genderless version exists too
                genderless_key = "_".join(key_from.split('_')[:-1]) # e.g. smplx_trans, smplx_betas, smplx_poses
                if genderless_key in list(file.keys()):
                    if key_from in keys_not_found:
                        keys_not_found.remove(key_from)
                    params = torch.tensor(file[genderless_key][::self.subsampling],
                                          dtype=torch.float32,
                                          device=self.device_loading)
                    if genderless_key.split('_')[1] == 'poses':
                        pass # params = self.rot_repr_conv(params) # Do this in __getitem__ function
                    if key_from in data:
                        data[key_from] = torch.vstack((data[key_from], params))
                    else:
                        data[key_from] = params

            for key_to in keys_to:
                if key_to not in list(file.keys()):
                    genderless_key = "_".join(key_to.split('_')[:-1])
                    if genderless_key not in list(file.keys()):
                        if self.allow_unsupervised:
                            print(f"INFO: Dataset does not contain key: {key_to}")
                            self.is_unsupervised = True
                        else:
                            keys_not_found.append(key_to)
                else:
                    params = torch.tensor(file[key_to][::self.subsampling],
                                          dtype=torch.float32,
                                          device=self.device_loading)
                    if key_to.split('_')[1] == 'poses':
                        pass # params = self.rot_repr_conv(params) # Do this in __getitem__
                    if key_to in data:
                        data[key_to] = torch.vstack((data[key_to], params))
                    else:
                        data[key_to] = params
                # Check whether genderless version exists too
                genderless_key = "_".join(key_to.split('_')[:-1])
                if genderless_key in list(file.keys()):
                    params = torch.tensor(file[genderless_key][::self.subsampling],
                                          dtype=torch.float32,
                                          device=self.device_loading)
                    if genderless_key.split('_')[1] == 'poses':
                        pass # params = self.rot_repr_conv(params) # Do this in __getitem__
                    if key_to in data:
                        data[key_to] = torch.vstack((data[key_to], params))
                    else:
                        data[key_to] = params
        if len(keys_not_found) != 0:

            raise ValueError(
                f"Expected dataset to contain keys: {keys_not_found}.\nIf this is only true for "
                "the target model type, consider setting allow_unsupervised=True"
            )
        return data, keys_from, keys_to


    def _create_normalizers(self,
                            normalization_method: str,
                            **kwargs
                            ) -> DotMap:
        normalizers = DotMap(_dynamic=False)    # _dynamic=False, otherwise IPython adds elements to it
        for key in self.keys_from:
            if key.split('_')[1] in self.parameters_to_learn:
                normalizers[key] = NORMALIATION_METHODS[normalization_method](**kwargs)
        return normalizers


    def fit_normalizers(self,
                         train_indices: Union[np.ndarray, List[int]]) -> None:
        if not hasattr(self, 'data'):
            raise ValueError("Normalizers can only be fitted _after_ the data is loaded.")
        for key, normalizer in self.normalizers.items():
            if not isinstance(normalizer, NORMALIATION_METHODS[self.normalization_method]):
                raise ValueError(f"Unexpected normalizer class: {normalizer.__class__}")
            normalizer.fit(self.data[key][train_indices])


    def get_normalizer_for_parameter(self, parameter) -> Union[Any, None]:
        key = f"{self.mode_from.to_internal_string()}_{parameter}_{self.gender}"
        if key in self.normalizers.keys():
            return self.normalizers[key]
        else:
            return None


    def get_normalizers(self) -> Union[DotMap, None]:
        return self.normalizers
