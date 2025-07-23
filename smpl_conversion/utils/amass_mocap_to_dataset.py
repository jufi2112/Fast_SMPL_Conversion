# Copyright (c) 2025 Julien Fischer, Chair of Computer Graphics and Visualization, TUD Dresden University of Technology.
# All rights reserved.
# Licensed under the MIT License. See LICENSE in the project root for license information.

import torch
import argparse
import numpy as np
import smpl_conversion

def amass_mocap_to_dataset(mocap_file: str,
                           output_file: str
                           ):
    """
        Converts the given mocap to a .npy file that can be loaded e.g. with
        the combined predictor

    Params
    ------
        mocap_file (str):
            Mocap file that should be transformed to a npy dataset
        output_file (str):
            File to which the dataset should be written
    """
    mocap = np.load(mocap_file)
    trans = mocap['trans']
    poses = mocap['poses']
    betas = np.repeat(mocap['betas'][None, ...], trans.shape[0], axis=0)
    data = np.hstack((trans, betas, poses))
    with open(output_file, "wb") as file:
        np.save(file, data)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Converts an AMASS mocap sequence to a .npy file")
    parser.add_argument('-i', '--input', type=str, required=True,
                        help="Input mocap sequence .npz file")
    parser.add_argument('-o', '--output', type=str, required=True,
                        help="Output file.")
    args = parser.parse_args()
    amass_mocap_to_dataset(args.input, args.output)