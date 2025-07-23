# Copyright (c) 2025 Julien Fischer, Chair of Computer Graphics and Visualization, TUD Dresden University of Technology.
# All rights reserved.
# Licensed under the MIT License. See LICENSE in the project root for license information.

import os
import sys
import shutil
import tarfile
import zipfile
import argparse
import numpy as np
from tqdm import tqdm
from os import path as osp
from typing import Dict, Union, List
from smpl_conversion.data.transforms import canonize_rotation_vectors_numpy

class AMASSSampler:
    def __init__(self,
                 amass_root_dir: str,
                 output_dir: str,
                 params_to_extract: List[str],
                 ):
        """
            Samples body model parameters from the different AMASS archives into
            the target directory.

            Params
            ------
                amass_root_dir (str):
                    Directory that contains smplh/ and smplx/ sub-directories
                output_dir (str):
                    Target output directory
                params_to_extract (list of str):
                    Parameters that should be extracted
        """
        if not osp.exists(amass_root_dir):
            raise ValueError(f"Input directory does not exist: {amass_root_dir}")
        if not osp.exists(output_dir):
            os.makedirs(output_dir)
        self.root_dir = amass_root_dir
        self.output_dir = output_dir
        self.keys_to_extract = params_to_extract


    def _betas_add_batch_dimension(self, betas: np.ndarray, n_poses: int) -> np.ndarray:
        """
            If necessary, adds a batch dimension corresponding to n_poses to the given betas

            Params
            ------
                betas (np.ndarray):
                    Shape components
                n_poses (int):
                    Batch size of pose parameters

            Returns
            -------
                np.ndarray: Shape parameters with batch dimension
        """
        if betas.ndim == 1:
            betas = np.expand_dims(betas, axis=0)
        if betas.shape[0] != 1:
            if betas.shape[0] != n_poses:
                raise ValueError(f"Cannot combine batch dimensions of betas {betas.shape[0]} and poses {n_poses}")
            else:
                return betas
        betas = np.repeat(betas, n_poses, axis=0)
        return betas


    def _adjust_number_shape_components(self,
                                        betas: np.ndarray,
                                        n_shape_components: int
                                        ) -> np.ndarray:
        """
            Adjusts the number of shape components to the given value. If more
            shape components are requested than are available, the betas are
            padded with trailing zeros.

            Params
            ------
                betas (np.ndarray):
                    Extracted beta paramerers of shape (B, n)
                n_shape_components (int):
                    Number of shape components that are to be used.

            Returns
            -------
                np.ndarray:
                    Adjusted betas with shape (B, n_shape_components)
        """
        if betas.shape[1] < n_shape_components:
            padding = np.zeros((betas.shape[0], n_shape_components - betas.shape[1]), dtype=betas.dtype)
            betas = np.hstack((betas, padding))
        else:
            betas = betas[:, :n_shape_components]
        return betas


    def _load_amass_file(self,
                         fpath: str,
                         model_type: str,
                         n_shape_components: int,
                         border_cutoff: float,
                         subsample_fps_target: int
                         ) -> Union[None, Dict[str, np.ndarray]]:
        """
            Loads data from the given file.

            Params
            ------
                fpath (str):
                    File path
                model_type (str):
                    Model type
                n_shape_components (int):
                    Number of shape components (betas) that should be extracted
                border_cuttoff (float):
                    Percentage of frames that should be cut from left and right border
                subsample_fps_target (int):
                    How many frames per second should be sampled
        """
        content = np.load(fpath)
        if "poses" not in list(content.keys()):
            return None
        gender = str(content['gender'].astype("U"))
        mocap_framerate = content['mocap_framerate'] if 'mocap_framerate' in list(content.keys()) else content['mocap_frame_rate']
        n_poses = content['poses'].shape[0]
        # Calculate sampling step size
        if subsample_fps_target == -1:
            step_size = 1
        else:
            step_size = int(mocap_framerate // subsample_fps_target)
        # calculate the element indices that should be sampled from the motion sequence
        cutoff_left = int(n_poses * border_cutoff)
        cutoff_right = int(n_poses * (1-border_cutoff))
        sampling_indices = list(range(cutoff_left, cutoff_right))[::step_size]
        n_samples = len(sampling_indices)
        data = {}
        for param in self.keys_to_extract:
            d = content[param]
            if param == "betas":
                d = self._betas_add_batch_dimension(d, n_samples)
                d = self._adjust_number_shape_components(d, n_shape_components)
            else:
                d = d[sampling_indices]
            if param == "poses":
                assert d.shape[1] == 156 if model_type == 'smplh' else 165, f"Pose dimension in file {fpath} does not match expectation: Is {d.shape[1]} and expected {156 if model_type == 'smplh' else 165}"
            key = f"{param}_{gender}"
            data[key] = d
        return data


    def _get_subdirs(self, root_dir: str) -> List[str]:
        """
            Returns all directories inside the given root_dir
        """
        subdirs = [
            element for element in os.listdir(root_dir) if osp.isdir(osp.join(root_dir, element))
        ]
        return subdirs


    def _save_extracted_data(self,
                             output_base_dir: str,
                             dataset_name: str,
                             model_name: str,
                             data: Dict[str, np.ndarray],
                             compress: bool = True
                             ):
        """
            Saves the given data to file.

            Params
            ------
                output_base_dir (str):
                    Base output directory
                dataset_name (str):
                    Name of the dataset
                model_name (str):
                    Name of the model to which the data belongs
                data (Dict[str, np.ndarray]):
                    Data that should be written to file
                compress (bool):
                    Whether written data should be compressed.
                    Defaults to True
        """
        os.makedirs(output_base_dir, exist_ok=True)
        model_dir = osp.join(output_base_dir, model_name)
        os.makedirs(model_dir, exist_ok=True)
        fname = f"{model_name}_{dataset_name}.npz"
        fpath = osp.join(model_dir, fname)
        if osp.exists(fpath):
            raise ValueError(f"File already exists at location {fpath}")
        if compress:
            np.savez_compressed(fpath, **data)
        else:
            np.savez(fpath, **data)
        return


    def _handle_moyo_dataset(self,
                             base_dir: str,
                             model_type: str,
                             n_shape_components: int,
                             border_cutoff: float,
                             subsample_fps_target: int,
                             compress: bool):
        data = {
            'train': None,
            'valid': None,
            'test': None,
            'extra': None
        }
        for subset in (pbar_moyo := tqdm(['train', 'val', 'test', 'extra'], position=2, leave=False)):
            pbar_moyo.set_description(f"MOYO Subset: {subset}")
            subset_path = osp.join(base_dir, subset)
            extracted_data = {}
            if not osp.exists(subset_path):
                raise ValueError(f"Could not find MOYO subset at {subset_path}")
            all_files = []
            for root, dirs, files in os.walk(subset_path):
                for file in files:
                    all_files.append(osp.join(root, file))
            for file in (pbar_moyo_file := tqdm(all_files, position=3, leave=False)):
                pbar_moyo_file.set_description(f"File: {osp.splitext(osp.basename(file))[0]}")
                data = self._load_amass_file(file, model_type, n_shape_components,
                                             border_cutoff, subsample_fps_target)
                if data is None:
                    tqdm.write(f"No pose attribute in {file}, skipping.")
                    continue
                for key_data, val in data.items():
                    key = f"{model_type}_{key_data}"
                    if key not in extracted_data:
                        extracted_data[key] = val
                    else:
                        extracted_data[key] = np.vstack((extracted_data[key], val))
            self._save_extracted_data(self.output_dir,
                                      f"MOYO_{'valid' if subset == 'val' else subset}",
                                      model_type,
                                      extracted_data,
                                      compress)


    def run(self,
            n_shape_components: int,
            border_cutoff: float,
            subsample_fps_target: int,
            compress_output: bool):
        """
            Extracts data from the AMASS datasets

            Params
            ------
                n_shape_components (int):
                    Number of shape (beta) components that should be extracted
                border_cutoff (float):
                    Percentage of data that should be ignored at the beginning
                    and end of motion capture sequences
                subsample_fps_target (int):
                    Target fps that extracted data should have
                compress_output (bool):
                    Whether the extracted data should be compressed
        """
        smplh_path = osp.join(self.root_dir, 'smplh')
        if not osp.exists(smplh_path):
            raise ValueError(f"The SMPL+H path does not exist: {smplh_path}")
        smplx_path = osp.join(self.root_dir, 'smplx')
        if not osp.exists(smplx_path):
            raise ValueError(f"The SMPL-X path does not exist: {smplx_path}")
        
        smplh_datasets = self._get_subdirs(smplh_path)
        smplx_datasets = self._get_subdirs(smplx_path)

        for datasets, base_path, model_name in (pbar_outer := tqdm(zip([smplh_datasets, smplx_datasets], [smplh_path, smplx_path], ['smplh', 'smplx']), position=0, leave=True)):
            pbar_outer.set_description(f"Model: {model_name.upper()}")
            # Iterate through all datasets for SMPL+H / SMPL-X
            for dataset in (pbar_middle := tqdm(datasets, position=1, leave=False)):
                pbar_middle.set_description(f"Dataset: {dataset}")
                dataset_path = osp.join(base_path, dataset)
                if dataset == 'MOYO':
                    self._handle_moyo_dataset(dataset_path, model_name, n_shape_components,
                                              border_cutoff, subsample_fps_target,
                                              compress_output)
                    continue
                sessions = [osp.join(dataset_path, element) for element in os.listdir(dataset_path) if osp.isdir(osp.join(dataset_path, element))]
                extracted_data = {}
                # Iterate through all sub-folders of this dataset
                for session in (pbar_inner := tqdm(sessions, position=2, leave=False)):
                    pbar_inner.set_description(f"Session: {osp.basename(session)}")
                    files = [osp.join(session, element) for element in os.listdir(session) if osp.splitext(element)[1] == '.npz']
                    # Iterate through all files in each sub-folder
                    for file in (pbar_files := tqdm(files, position=3, leave=False)):
                        pbar_files.set_description(f"File: {osp.basename(file)}")
                        data = self._load_amass_file(file, model_name, n_shape_components,
                                                     border_cutoff, subsample_fps_target)
                        if data is None:
                            fname = osp.splitext(osp.basename(file))[0]
                            if not 'shape' in fname and not fname.split('_')[-1] == 'stagei':
                                tqdm.write(f"No pose attribute in {file}, skipping.")
                            continue
                        for key_data, val in data.items():
                            key = f"{model_name}_{key_data}"
                            if key not in extracted_data:
                                extracted_data[key] = val
                            else:
                                extracted_data[key] = np.vstack((extracted_data[key], val))
                # Calculate canonical pose parameters
                for gender in ['male', 'female']:
                    key = f"{model_name}_poses_{gender}"
                    poses = extracted_data.get(key, None)
                    if poses is None:
                        continue
                    extracted_data[key] = canonize_rotation_vectors_numpy(poses)
                # Save extracted data
                self._save_extracted_data(self.output_dir,
                                          dataset,
                                          model_name,
                                          extracted_data,
                                          compress_output)



class AMASSArchiveExtractor:
    def __init__(self, root_dir: str):
        """
            Extracts the AMASS sub datasets

            Params
            ------
                root_dir (str):
                    Path where the AMASS archives are located
        """
        self.root_dir = root_dir


    def _validate_paths(self, smplh_path: str, smplx_path: str):
        if not osp.isdir(smplh_path):
            raise ValueError(f"SMPL+H path does not exist: {smplh_path}")
        if not osp.isdir(smplx_path):
            raise ValueError(f"SMPL-X path does not exist: {smplx_path}")


    def _extract_bz2_archive(self,
                             archive_path: str,
                             base_dir: str):
        """
            Extracts the given archive to the given directory.

            Params
            ------
                archive_path (str):
                    Path to the archive that should be extracted
                base_dir (str):
                    Directory where the archive should be extracted to.
        """
        tmp_dir = osp.join(base_dir, 'tmp')
        if osp.exists(tmp_dir):
            shutil.rmtree(tmp_dir)
        with tarfile.open(archive_path, "r:bz2") as tar:
            tar.extractall(path=tmp_dir)
        extracted_dirs = [osp.join(tmp_dir, element) for element in os.listdir(tmp_dir) if osp.isdir(osp.join(tmp_dir, element))]
        if len(extracted_dirs) > 1:
            tqdm.write(f"Extracted multiple top-level directories for archive {osp.splitext(osp.basename(archive_path))[0]}")
        for directory in extracted_dirs:
            new_location = osp.dirname(osp.dirname(directory))
            new_fpath = osp.join(new_location, osp.basename(directory))
            shutil.move(directory, new_fpath)
        shutil.rmtree(tmp_dir)


    def _extract_zip_archive(self,
                             archive_path: str,
                             archive_name: str,
                             base_dir: str):
        """
            Extracts the given archive to the given directory.

            Params
            ------
                archive_path (str):
                    Path to the archive that should be extracted
                archive_name (str):
                    Name of the archive. Used for special treatment of MOYO dataset
                base_dir (str):
                    Directory where the archive should be extracted to.
        """
        with zipfile.ZipFile(archive_path, 'r') as zip_ref:
            zip_ref.extractall(path=osp.join(base_dir, archive_name))


    def extract_datasets(self):
        smplh_path = osp.join(self.root_dir, 'smplh')
        smplx_path = osp.join(self.root_dir, 'smplx')
        self._validate_paths(smplh_path, smplx_path)

        for base_dir, name in zip([smplh_path, smplx_path], ['SMPL+H', 'SMPL-X']):
            tqdm.write(f"Extracting sub-datasets for model type {name}")
            archives = [osp.join(base_dir, element) for element in os.listdir(base_dir) if osp.splitext(element)[1] in ['.bz2', '.zip']]
            for archive in (pbar := tqdm(archives)):
                archive_name = osp.splitext(osp.splitext(osp.basename(archive))[0])[0].split('_')[0]
                pbar.set_description(f"Dataset {archive_name}")
                if osp.splitext(archive)[1] == '.bz2':
                    self._extract_bz2_archive(archive, base_dir)
                elif osp.splitext(archive)[1] == '.zip':
                    self._extract_zip_archive(archive, archive_name, base_dir)
                else:
                    tqdm.write(f"Unsupported file format: {archive}")
                    continue



def combine_subdatasets(root_dir: str,
                        dataset_splits: Dict[str, str],
                        save_compressed: bool):
    """
        Combines sub-datasets into training, validation, and test datasets

        Params
        ------
            root_dir (str):
                Root directory that contains extracted smplx and smplh directories
            dataset_splits (dict):
                Dictionary that defines which sub-datasets are to be used for which
                purposes.
            save_compressed (bool):
                Whether datasets should be saved compressed.
    """
    if not osp.exists(root_dir):
        raise ValueError(f"Root directory does not exist: {root_dir}")
    smplh_dir = osp.join(root_dir, 'smplh')
    if not osp.exists(smplh_dir):
        raise ValueError(f"SMPL+H directory does not exist: {smplh_dir}")
    smplx_dir = osp.join(root_dir, 'smplx')
    if not osp.exists(smplx_dir):
        raise ValueError(f"SMPL-X directory does not exist: {smplx_dir}")
    genders = ['male', 'female']
    sizes = {
        'smplx': {
            mode: {
                gender: 0
                for gender in genders
            }
            for mode in dataset_splits.keys()
        },
        'smplh': {
            mode: {
                gender: 0
                for gender in genders
            }
            for mode in dataset_splits.keys()
        }
    }
    dataset_modes = list(dataset_splits.keys())
    for subdir, model_name in (pbar_outer := tqdm(zip([smplh_dir, smplx_dir], ['smplh', 'smplx']), position=0, leave=True)):
        pbar_outer.set_description(f"Model type: {model_name.upper()}")
        for mode in (pbar_middle := tqdm(dataset_modes, position=1, leave=False)):
            pbar_middle.set_description(f"Mode: {mode}")
            subsets = dataset_splits[mode]
            data_combined = {}
            for subset in (pbar_inner := tqdm(subsets, position=2, leave=False)):
                pbar_inner.set_description(f"Subset {subset}")
                expected_name = f"{model_name}_{subset}.npz"
                fpath = osp.join(subdir, expected_name)
                if not osp.exists(fpath):
                    tqdm.write(f"Did not find subset {subset} for {model_name.upper()} {mode}")
                    continue
                data = np.load(fpath)
                for key, val in data.items():
                    if key not in data_combined.keys():
                        data_combined[key] = val
                    else:
                        data_combined[key] = np.vstack((data_combined[key], val))
            for key, val in data_combined.items():
                model, param, gender = key.split('_')
                if param != 'poses':
                    continue
                if sizes[model][mode][gender] == 0:
                    sizes[model][mode][gender] = val.shape[0]

            dataset_fname = f"{model_name}_{mode}.npz"
            savepath = osp.join(root_dir, dataset_fname)
            if save_compressed:
                np.savez_compressed(savepath, **data_combined)
            else:
                np.savez(savepath, **data_combined)
    print("Extracted the following number of elements for each dataset")
    for model, per_dataset in sizes.items():
        print(f"{model.upper()}")
        for dataset, per_gender in per_dataset.items():
            print(f"   {dataset.upper()}")
            for gender, elements in per_gender.items():
                print(f"      {gender.upper()}: {elements}")



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Samples body model parameters from the different AMASS archieves.")
    parser.add_argument('-i', '--input-amass', type=str, required=True,
                        help="Path to the root directory that contains smplh/ and smplx/ directories")
    parser.add_argument('-o', '--output', type=str, required=True,
                        help="Output directory where extracted parameters should be written to.")
    parser.add_argument('--skip-extraction', action='store_true',
                        help="Skip archive extraction step")
    parser.add_argument('--skip-combine', action='store_true',
                        help="Skip combination of sub-dataset into single files")
    args = parser.parse_args()
    # Step 1: Extract datasets from downloaded files
    if not args.skip_extraction:
        extractor = AMASSArchiveExtractor(args.input_amass)
        extractor.extract_datasets()
    # Step 2: For each sub-dataset, combine motion sequences of different
    #         subjects into a single .npz file
    keys_to_extract = ['trans', 'betas', 'poses']
    border_cutoff = 0.1
    subsample_fps_target = 5 # -1 ... Use all available frames
    compress_output = True
    n_shape_components = 16
    if not args.skip_combine:
        sampler = AMASSSampler(args.input_amass,
                               args.output,
                               keys_to_extract)
        tqdm.write("Compiling sub-datasets into single files.")
        sampler.run(n_shape_components,
                    border_cutoff,
                    subsample_fps_target,
                    compress_output)
    # Step 3: Combine different sub-datasets into train, validation, and test data
    # Taken from AMASS' recommendation
    dataset_splits = {
        'validation': [
            'HumanEva',
            'MPI_HDM05', 'HDM05',
            'SFU',
            'MPI_mosh', 'MoSH',
            'DFaust_67', 'DFaust',
            'MOYO_valid'
        ],
        'test': [
            'Transitions_mocap', 'Transitions',
            'SSM_synced', 'SSM',
            'SOMA',
            'DanceDB',
            'MOYO_test'
        ],
        'train': [
            'CMU',
            'MPI_Limits', 'PosePrior',
            'TotalCapture',
            'Eyes_Japan_Dataset',
            'KIT',
            'BMLhandball', # SMPL+H only
            'BMLmovi',
            'BioMotionLab_NTroje', 'BMLrub',
            'EKUT',
            'TCD_handMocap', 'TCDHands',
            'ACCAD',
            'WEIZMANN', # SMPL-X only
            'HUMAN4D',
            'GRAB',
            'CNRS', # SMPL-X only
            'MOYO_train',
            'MOYO_extra'
        ]
    }
    tqdm.write("Creating files for train, validation, and test datasets")
    combine_subdatasets(args.output,
                        dataset_splits,
                        compress_output)
    tqdm.write(f"Finished, resulting datasets are located at {args.output}")
