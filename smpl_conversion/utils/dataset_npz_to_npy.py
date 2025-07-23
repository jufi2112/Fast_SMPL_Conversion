# Copyright (c) 2025 Julien Fischer, Chair of Computer Graphics and Visualization, TUD Dresden University of Technology.
# All rights reserved.
# Licensed under the MIT License. See LICENSE in the project root for license information.

import os
import argparse
import numpy as np
from os import path as osp

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Converts npz dataset to npy dataset for use in combined predictor")
    parser.add_argument('-i', '--input', type=str, required=True,
                        help="Path to the input npz dataset that should be converted. Assumes this dataset has standard naming convention")
    parser.add_argument('--output-dir', type=str, default=None,
                        help="Output directory. Optional, defaults to same directory as input")
    args = parser.parse_args()
    if not osp.exists(args.input):
        raise ValueError(f"Could not find provided .npz dataset: {args.input}")
    if osp.splitext(args.input)[1] != '.npz':
        raise ValueError(f"Given npz dataset is not a valid .npz file: {args.input}")
    output_path = args.output_dir
    output_fname = osp.splitext(osp.basename(args.input))[0]
    if output_path is None:
        output_path = osp.dirname(args.input)
    os.makedirs(output_path, exist_ok=True)

    # Load npz dataset
    input_data = np.load(args.input)
    input_keys = list(input_data.keys())
    model_type = input_keys[0].split('_')[0]
    genders = ['male', 'female']
    parameters = ['trans', 'betas', 'poses']
    gendered_keys = [f'{model_type}_{param}_{gender}' for param in parameters for gender in genders]
    
    for gender in genders:
        npy_data_dict = {
            'trans': None,
            'betas': None,
            'poses': None
        }
        for param in parameters:
            # Check for gendered data
            key = f"{model_type}_{param}_{gender}"
            if key in input_data:
                if npy_data_dict[param] is None:
                    npy_data_dict[param] = input_data[key]
                else:
                    npy_data_dict[param] = np.vstack((npy_data_dict[param], input_data[key]))
            ungendered_key = f"{model_type}_{param}"
            # check for ungendered data
            if ungendered_key in input_data:
                if npy_data_dict[param] is None:
                    npy_data_dict[param] = input_data[ungendered_key]
                else:
                    npy_data_dict[param] = np.vstack((npy_data_dict[param], input_data[ungendered_key]))
        # Save as gender specific npy file
        data = np.hstack((npy_data_dict['trans'], npy_data_dict['betas'], npy_data_dict['poses']))
        bs = len(data)
        print(f"Found {bs} entries for gender {gender}", flush=True)
        np.save(osp.join(output_path, f"{output_fname}_{gender}"), data)
