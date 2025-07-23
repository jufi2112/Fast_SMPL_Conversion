# Copyright (c) 2025 Julien Fischer, Chair of Computer Graphics and Visualization, TUD Dresden University of Technology.
# All rights reserved.
# Licensed under the MIT License. See LICENSE in the project root for license information.

import os
import pickle
import argparse
import numpy as np
from tqdm import tqdm
from os import path as osp
from smpl_conversion.data.transforms import canonize_rotation_vectors_numpy
from smpl_conversion.utils.motion_x import align_motion_x_to_amass

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Processes 3DPW data")
    parser.add_argument('-i', '--input', type=str, required=True,
                        help="3DPW input directory. Usually named 'sequenceFiles'")
    parser.add_argument('-o', '--output', type=str, required=True,
                        help="Output path of the processed datasets")
    parser.add_argument('--n-shape-comps', type=int, default=16,
                        help="Number of shape components to extract. Optional, defaults to 16")
    parser.add_argument('--split', action='store_true', help="This flag indicates "
                        "that 3DPW data should be split into train, validation, and test data")
    parser.add_argument('--transform-to-amass', action="store_true",
                        help="Flag indicates that translation and global orientation "
                        "parameters should be converted to AMASS' coordinate system")
    args = parser.parse_args()
    if not osp.isdir(args.input):
        raise ValueError(f"Provided 3DPW input path is invalid: {args.input}")
    output_dir = args.output
    os.makedirs(output_dir, exist_ok=True)
    n_shape_params = args.n_shape_comps
    all_data = {
        'train': {},
        'validation': {},
        'test': {}
    }

    # Iterate over the different subsets
    for subset in (pbar_subset := tqdm(['train', 'validation', 'test'], position=0, leave=True)):
        pbar_subset.set_description(f"Subset {subset}")
        subset_path = osp.join(args.input, subset)
        if not osp.exists(subset_path):
            raise ValueError(f"Subset {subset} could not be found at {subset_path}")
        elements = [elem for elem in os.listdir(subset_path) if osp.splitext(elem)[1] == '.pkl']
        # Fetch data from the pickle files
        for element in elements:
            elem_path = osp.join(subset_path, element)
            with open(elem_path, 'rb') as file:
                data = pickle.load(file, encoding='latin1')
                # Each attribute is a list with an entry for each person in the scene
                genders = data['genders']
                assert isinstance(genders, list)
                for p_idx in range(len(genders)):
                    assert genders[p_idx] in ['m', 'f']
                    gender = 'male' if genders[p_idx] == 'm' else 'female'
                    person_data = {}
                    for param, standardized_name in zip(['trans_60Hz', 'poses_60Hz', 'betas'], ['trans', 'poses', 'betas']):
                        # Key for overall dataset
                        key = f"smpl_{standardized_name}_{gender}"
                        assert isinstance(data[param], list)
                        d = data[param][p_idx]
                        # d should now be a numpy array
                        assert isinstance(d, np.ndarray)
                        # If not batch dimension, add it
                        if d.ndim == 1:
                            d = d[None, :]
                        if param == 'betas':
                            if d.shape[1] >= n_shape_params:
                                d = d[:, :n_shape_params]
                            elif d.shape[1] < n_shape_params:
                                d = np.hstack((d, np.zeros((d.shape[0], n_shape_params - d.shape[1]),
                                                           dtype=np.float32)))
                            if d.shape[0] == 1:
                                # match batch size of translation and poses
                                # translation and pose keys will be processed before shape, so should already be present in person_data dict
                                d = np.tile(d, (person_data[f'smpl_trans_{gender}'].shape[0], 1))
                        person_data[key] = d
                    # Add person's data to overall dataset
                    for key, val in person_data.items():
                        if key not in all_data[subset]:
                            all_data[subset][key] = val
                        else:
                            all_data[subset][key] = np.vstack((all_data[subset][key], val))

    # Post-processing of pose and translation parameters
    for subset in all_data.keys():
        for key in ['smpl_poses_male', 'smpl_poses_female']:
            if key not in all_data[subset]:
                print(f"\nDid not find key {key} for subset {subset}\n", flush=True)
                continue
            # Canonize rotation vectors
            canon_poses = canonize_rotation_vectors_numpy(all_data[subset][key])
            all_data[subset][key] = canon_poses
        # if requested, transform 3DPW from OpenGL coordinate system to Blender coordinate system
        if args.transform_to_amass:
            for gender in ['male', 'female']:
                key_trans = f"smpl_trans_{gender}"
                key_poses = f"smpl_poses_{gender}"
                if key_trans not in all_data[subset] or key_poses not in all_data[subset]:
                    print(f"\nDid not find keys {key_trans} or {key_poses} in subset {subset}\n")
                    continue
                go = all_data[subset][key_poses][:, :3]
                trans = all_data[subset][key_trans]
                go_aligned, trans_aligned = align_motion_x_to_amass(go, trans)
                all_data[subset][key_poses][:, :3] = go_aligned
                all_data[subset][key_trans] = trans_aligned

    merged = {}
    # Only keep datasets split if explicitly told by user
    for subset in all_data.keys():
        if args.split:
            # Keep datasets split
            fpath = osp.join(output_dir, f'smpl_{subset}_3dpw_{"aligned" if args.transform_to_amass else "unaligned"}.npz')
            np.savez(fpath, **(all_data[subset]))
        else:
            for key, val in all_data[subset].items():
                if key in merged:
                    merged[key] = np.vstack((merged[key], val))
                else:
                    merged[key] = val
    if not args.split:
        fpath = osp.join(output_dir, f'smpl_3dpw_{"aligned" if args.transform_to_amass else "unaligned"}.npz')
        np.savez(fpath, **merged)
    print("Finished")
