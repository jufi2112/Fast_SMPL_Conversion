# Copyright (c) 2025 Julien Fischer, Chair of Computer Graphics and Visualization, TUD Dresden University of Technology.
# All rights reserved.
# Licensed under the MIT License. See LICENSE in the project root for license information.

import os
import argparse
import numpy as np
from tqdm import tqdm
from os import path as osp

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Merges AMASS and Motion-X datasets")
    parser.add_argument('--input-amass', type=str, required=True,
                        help="Path to the directory that contains AMASS train, validation, and test datasets")
    parser.add_argument('--input-motionx', type=str, required=True,
                        help="Path to the directory that contains Motion-X train, validation, and test datasets")
    parser.add_argument('-o', '--output', type=str, required=True,
                        help="Output directory")
    args = parser.parse_args()
    if not osp.exists(args.input_amass):
        raise ValueError(f"Provided AMASS directory does not exist: {args.input_amass}")
    if not osp.exists(args.input_motionx):
        raise ValueError(f"Provided Motion-X directory does not exist: {args.input_motionx}")
    os.makedirs(args.output, exist_ok=True)
    amass_files = {
        'train': 'smplx_train.npz',
        'validation': 'smplx_validation.npz',
        'test': 'smplx_test.npz'
    }
    motionx_files = {
        'train': 'smplx_train_motion-x.npz',
        'validation': 'smplx_validation_motion-x.npz',
        'test': 'smplx_test_motion-x.npz'
    }
    # Search for AMASS and Motion-X files
    for mode in tqdm(['train', 'validation', 'test'], desc="Datasets"):
        data = {}
        amass_fpath = osp.join(args.input_amass, amass_files[mode])
        if not osp.exists(amass_fpath):
            raise ValueError(f"Could not find AMASS {mode} dataset at {amass_fpath}")
        motionx_fpath = osp.join(args.input_motionx, motionx_files[mode])
        if not osp.exists(motionx_fpath):
            raise ValueError(f"Could not find Motion-X {mode} dataset at {motionx_fpath}")
        amass = np.load(amass_fpath)
        motionx = np.load(motionx_fpath)
        for key, value in amass.items():
            if key in data:
                data[key] = np.vstack((data[key], value))
            else:
                data[key] = value
        for key, value in motionx.items():
            if key in data:
                data[key] = np.vstack((data[key], value))
            else:
                data[key] = value
        np.savez(osp.join(args.output, f'smplx_{mode}_amass_motionx.npz'), **data)
