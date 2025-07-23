# Copyright (c) 2025 Julien Fischer, Chair of Computer Graphics and Visualization, TUD Dresden University of Technology.
# All rights reserved.
# Licensed under the MIT License. See LICENSE in the project root for license information.

import os
import argparse
import numpy as np
from os import path as osp
from smpl_conversion.utils.model_infos import MODEL_STATS
from smpl_conversion.utils.enum_configurations import PoseRepresentation, BodyModelType

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Converts a .npy dataset file into a .npz dataset file with standard naming scheme")
    parser.add_argument('--input-male', type=str, required=True,
                        help="Input npy file with male parameters")
    parser.add_argument('--input-female', type=str, required=True,
                        help="Input npy file with female parameters")
    parser.add_argument('-m', '--model-type', type=str, required=True,
                        help="Model type of the parameters")
    parser.add_argument('-n', '--name', type=str, required=True,
                        help="Name of the created npz dataset")
    parser.add_argument('--shape-components', type=int, default=16,
                        help="Number of shape components. Optional, defaults to 16")
    parser.add_argument('--rot_rep', type=str, default='rot_vec',
                        help="Rotation representation of pose parameters. Optional, defaults to rot_vec")
    parser.add_argument('--output-dir', type=str, default=None,
                        help="Output directory. Optional, defaults to same directory as male input npy file")
    args = parser.parse_args()
    if not osp.exists(args.input_male) or osp.splitext(args.input_male)[1] != '.npy':
        raise ValueError(f"Invalid male npy input file: {args.input_male}")
    if not osp.exists(args.input_female) or osp.splitext(args.input_female)[1] != '.npy':
        raise ValueError(f"Invalid female npy input file: {args.input_female}")
    rot_rep = PoseRepresentation.from_string(args.rot_rep)
    if rot_rep is None:
        raise ValueError(f"Invalid rotation representation: {args.rot_rep}")
    output_dir = args.output_dir
    if output_dir is None:
        output_dir = osp.dirname(args.input_male)
    os.makedirs(output_dir, exist_ok=True)
    output_fpath = osp.join(output_dir, args.name)

    # Load male and female datasets
    data_male = np.load(args.input_male)
    data_female = np.load(args.input_female)
    # Split into subsets
    data = {}
    # Translation
    data[f"{args.model_type}_trans_male"] = data_male[:, :3]
    data[f"{args.model_type}_trans_female"] = data_female[:, :3]
    # Shape
    data[f"{args.model_type}_betas_male"] = data_male[:, 3:3+args.shape_components]
    data[f"{args.model_type}_betas_female"] = data_female[:, 3:3+args.shape_components]
    # Pose
    data[f"{args.model_type}_poses_male"] = data_male[:, 3+args.shape_components:]
    assert data[f"{args.model_type}_poses_male"].shape[1] == MODEL_STATS[BodyModelType.from_string(args.model_type)]['joints'] * rot_rep.get_number_components(), "Invalid number of male pose parameters"
    data[f"{args.model_type}_poses_female"] = data_female[:, 3+args.shape_components:]
    assert data[f"{args.model_type}_poses_female"].shape[1] == MODEL_STATS[BodyModelType.from_string(args.model_type)]['joints'] * rot_rep.get_number_components(), "Invalid number of female pose parameters"
    # Save as npz file
    np.savez(output_fpath, **data)
