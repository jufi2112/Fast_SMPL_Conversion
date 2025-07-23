# Copyright (c) 2025 Julien Fischer, Chair of Computer Graphics and Visualization, TUD Dresden University of Technology.
# All rights reserved.
# Licensed under the MIT License. See LICENSE in the project root for license information.

import os
import pickle
import argparse
import numpy as np
from tqdm import tqdm
from os import path as osp
from smpl_conversion.utils.motion_x import align_motion_x_to_amass
from smpl_conversion.data.transforms import canonize_rotation_vectors_numpy

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Assemble AGORA dataset")
    parser.add_argument('-i', '--input', type=str, required=True,
                        help="Base directory of AGORA (usually called 'smplx_gt')")
    parser.add_argument('-o', '--output', type=str, required=True,
                        help="Output directory for assembled AGORA dataset")
    parser.add_argument('--n-shape-params', type=int, default=16,
                        help="Number of shape components")
    parser.add_argument('--split', action="store_true",
                        help="Indicates that separate datasets should be saved for train and validation")
    parser.add_argument('--transform-to-amass', action="store_true",
                        help="Indicates that AGORA data should be transformed to AMASS' coordinate system")
    args = parser.parse_args()
    if not osp.exists(args.input):
        raise ValueError(f'Provided AGORA input path does not exist: {args.input}')
    os.makedirs(args.output, exist_ok=True)
    subsets = [element for element in os.listdir(args.input) if osp.isdir(osp.join(args.input, element)) and 'adults' in element.split('_')]
    data = {
        'train': {},
        'validation': {}
    }
    for subset in tqdm(subsets, position=0, leave=True, desc="Subset processing"):
        subset_fpath = osp.join(args.input, subset)
        dataset_mode = 'validation'
        if subset.split('_')[0] == 'trainset':
            dataset_mode = 'train'
        pkl_files = [file for file in os.listdir(subset_fpath) if osp.splitext(file)[1] == '.pkl']
        subset_params = {}
        for param_file in pkl_files:
            fpath = osp.join(subset_fpath, param_file)
            with open(fpath, 'rb') as file:
                d = pickle.load(file)
                try:
                    gender = d['gender']
                except KeyError:
                    print(f"Could not determine gender for file {fpath}")
                    continue
                trans = d['transl']
                betas = d['betas']
                if betas.shape[1] >= args.n_shape_params:
                    betas = betas[:, :args.n_shape_params]
                else:
                    betas = np.hstack((betas, np.zeros((len(betas), args.n_shape_params-betas.shape[1]), dtype=np.float32)))
                go = d['global_orient']
                if args.transform_to_amass:
                    go, trans = align_motion_x_to_amass(go, trans)
                poses = np.hstack((go,
                                   d['body_pose'],
                                   d['jaw_pose'],
                                   d['leye_pose'],
                                   d['reye_pose'],
                                   d['left_hand_pose'],
                                   d['right_hand_pose']))
                poses_canon = canonize_rotation_vectors_numpy(poses)
                for param_name, param in zip(['trans', 'betas', 'poses'], [trans, betas, poses_canon]):
                    key = f"smplx_{param_name}_{gender}"
                    if key not in subset_params:
                        subset_params[key] = param
                    else:
                        subset_params[key] = np.vstack((subset_params[key], param))
        for key, params in subset_params.items():
            if key not in data[dataset_mode]:
                data[dataset_mode][key] = params
            else:
                data[dataset_mode][key] = np.vstack((data[dataset_mode][key], params))
    if args.split:
        for mode, params in data.items():
            dataset_name = f"smplx_{mode}_agora_{'aligned' if args.transform_to_amass else 'unaligned'}.npz"
            np.savez(osp.join(args.output, dataset_name), **params)
    else:
        merged = {}
        for mode in data:
            for key, val in data[mode].items():
                if key in merged:
                    merged[key] = np.vstack((merged[key], val))
                else:
                    merged[key] = val
        fname = f'smplx_agora_{"aligned" if args.transform_to_amass else "unaligned"}.npz'
        np.savez(osp.join(args.output, fname), **merged)
