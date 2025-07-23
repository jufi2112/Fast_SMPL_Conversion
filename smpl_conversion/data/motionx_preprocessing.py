# Copyright (c) 2025 Julien Fischer, Chair of Computer Graphics and Visualization, TUD Dresden University of Technology.
# All rights reserved.
# Licensed under the MIT License. See LICENSE in the project root for license information.

import os
import argparse
import numpy as np
from tqdm import tqdm
from os import path as osp
from typing import Dict, Union, List
from smpl_conversion.data.transforms import canonize_rotation_vectors_numpy
from smpl_conversion.utils.motion_x import parse_motion_x_322_file, align_motion_x_to_amass

class MotionXAccumulator:
    def __init__(self,
                 base_dir: str,
                 output_dir: str,
                 n_shape_params: int
                 ):
        """
            Accumulates parameters of the same motion-x subset into a single
            .npz file

            Params
            ------
                base_dir (str):
                    Path to the base directory (usually 'smplx_322')
                output_dir (str):
                    Path to the output directory. If it does not exist, it will
                    be created
                n_shape_params (int):
                    Number of shape parameters that the dataset should contain
        """
        if not osp.exists(base_dir):
            raise ValueError(f"Provided base dir does not exist: {base_dir}")
        os.makedirs(output_dir, exist_ok=True)
        if not osp.exists(output_dir):
            raise ValueError(f"Provided output dir does not exist and could not be created: {output_dir}")
        self.base_dir = base_dir
        self.output_dir = output_dir
        self.n_shape_params = n_shape_params


    def run(self,
            subsample_step_size: int):
        """
            Accumulates the different Motion-X subdatasets into single .npz files.

            Params
            ------
                subsample_step_size (int):
                    Subsampling step size.
        """
        datasets = [element for element in os.listdir(self.base_dir) if osp.isdir(osp.join(self.base_dir, element))]
        for dataset in (pbar_dataset := tqdm(datasets, position=0, leave=True)):
            pbar_dataset.set_description(f"Processing dataset {dataset}")
            dataset_path = osp.join(self.base_dir, dataset)
            subdirs = [element for element in os.listdir(dataset_path) if osp.isdir(osp.join(dataset_path, element))]
            data = {'smplx_trans': None, 'smplx_betas': None, 'smplx_poses': None}
            for subdir in (pbar_subdir := tqdm(subdirs, position=1, leave=False)):
                pbar_subdir.set_description(f"Processing subdir {subdir}")
                subdir_path = osp.join(dataset_path, subdir)
                files = [file for file in os.listdir(subdir_path) if osp.isfile(osp.join(subdir_path, file)) and osp.splitext(file)[1] == '.npy']
                for file in (pbar_files := tqdm(files, position=2, leave=False)):
                    pbar_files.set_description("Processing files")
                    fpath = osp.join(subdir_path, file)
                    d = np.load(fpath)
                    parsed_d: Dict = parse_motion_x_322_file(d,
                                                             self.n_shape_params,
                                                             True,
                                                             subsample_step_size)
                    if data['smplx_trans'] is None:
                        data['smplx_trans'] = parsed_d['trans']
                    else:
                        data['smplx_trans'] = np.vstack((data['smplx_trans'], parsed_d['trans']))

                    if data['smplx_betas'] is None:
                        data['smplx_betas'] = parsed_d['betas']
                    else:
                        data['smplx_betas'] = np.vstack((data['smplx_betas'], parsed_d['betas']))

                    if data['smplx_poses'] is None:
                        data['smplx_poses'] = parsed_d['poses']
                    else:
                        data['smplx_poses'] = np.vstack((data['smplx_poses'], parsed_d['poses']))
            # Finished accumulating all parameters of this dataset
            poses = data['smplx_poses']
            data['smplx_poses'] = canonize_rotation_vectors_numpy(poses)
            np.savez(osp.join(self.output_dir, dataset+'.npz'), **data)
        print("Finished sub-dataset accumulation!")



class MotionXCombiner:
    def __init__(self,
                 base_dir: str,
                 output_dir: str
                 ):
        """
            Combines multiple Motion-X subdatasets into training, validation, and test sets

            Params
            ------
                base_dir (str):
                    Directory where separate dataset files are located
                output_dir (str):
                    Directory where combined datasets should be written to
        """
        if not osp.exists(base_dir) or not osp.isdir(base_dir):
            raise ValueError(f"Provided base directory {base_dir} does not exist!")
        if not osp.exists(output_dir):
            os.makedirs(output_dir)
        self.base_dir = base_dir
        self.output_dir = output_dir


    def run(self,
            dataset_splits: (Dict[str, str]),
            align_to_amass: bool
            ):
        """
            Performs the combination

            Params
            ------
                dataset_splits (dict):
                    Dictionary that contains the train, valid, and test splits
                align_to_amass (bool):
                    Whether the Motion-X data should be aligned to AMASS' coordinate system
        """
        for dataset_type in (pbar_dataset_types := tqdm(['train', 'validation', 'test'], position=0, leave=True)):
            pbar_dataset_types.set_description(f"Combining {dataset_type} dataset")
            datasets = dataset_splits[dataset_type]
            data = {'smplx_trans': None, 'smplx_betas': None, 'smplx_poses': None}
            for dataset in (pbar_dataset := tqdm(datasets, position=1, leave=False)):
                pbar_dataset.set_description(f"Fetching data from {dataset}")
                fname = dataset + '.npz'
                fpath = osp.join(self.base_dir, fname)
                if not osp.exists(fpath):
                    raise ValueError(f"Could not find dataset {dataset} at {fpath}")
                set_data = np.load(fpath)
                trans = set_data['smplx_trans']
                betas = set_data['smplx_betas']
                poses = set_data['smplx_poses']
                go = poses[:, :3]
                if align_to_amass:
                    go, trans = align_motion_x_to_amass(go, trans)
                    poses[:, :3] = go
                if data['smplx_trans'] is None:
                    data['smplx_trans'] = trans
                else:
                    data['smplx_trans'] = np.vstack((data['smplx_trans'], trans))
                if data['smplx_betas'] is None:
                    data['smplx_betas'] = betas
                else:
                    data['smplx_betas'] = np.vstack((data['smplx_betas'], betas))
                if data['smplx_poses'] is None:
                    data['smplx_poses'] = poses
                else:
                    data['smplx_poses'] = np.vstack((data['smplx_poses'], poses))
            fname_dataset = f'smplx_{dataset_type}_motion-x_{"aligned" if align_to_amass else "unaligned"}.npz'
            fpath_dataset = osp.join(self.output_dir, fname_dataset)
            np.savez(fpath_dataset, **data)
        print("Finished dataset combining!")



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Accumulates motion-x subdatasets into single files")
    parser.add_argument('-i', '--input', type=str, required=True,
                        help="Path to the motion-x base dir (usually 'smplx_322')")
    parser.add_argument('-o', '--output', type=str, required=True,
                        help="Output directory")
    parser.add_argument('--skip-accumulation', action='store_true',
                        help="Skip dataset accumulation step")
    args = parser.parse_args()
    n_shape_params = 16
    subsample_step_size = 10
    if not args.skip_accumulation:
        accumulator = MotionXAccumulator(args.input, args.output, n_shape_params)
        accumulator.run(subsample_step_size=subsample_step_size)
    # Align to AMASS and combine into train, validation and test set
    datasets = {
        'train': [
            'aist',
            'fitness',
            'game_motion',
            'HAA500',
            'humman',
            'idea400',
            'music'
        ],
        'validation': [
            'animation',
            'kungfu'
        ],
        'test': [
            'dance',
            'perform'
        ]
    }
    align_to_amass = True
    combiner = MotionXCombiner(args.output, args.output)
    combiner.run(datasets, align_to_amass)
