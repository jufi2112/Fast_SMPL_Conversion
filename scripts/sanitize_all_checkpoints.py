# Copyright (c) 2025 Julien Fischer, Chair of Computer Graphics and Visualization, TUD Dresden University of Technology.
# All rights reserved.
# Licensed under the MIT License. See LICENSE in the project root for license information.

import os
import argparse

from tqdm import tqdm
from os import path as osp
from smpl_conversion.utils.checkpoint import sanitize_checkpoint_for_distribution

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Sanitizes all checkpoints in the given base directory")
    parser.add_argument('-b', '--base-dir', type=str, required=True,
                        help="Path to the base directory")
    args = parser.parse_args()
    if not osp.exists(args.base_dir):
        raise ValueError(f"Provided base directory does not exist: {args.base_dir}")
    conversion_dirs = [osp.join(args.base_dir, d) for d in os.listdir(args.base_dir) if osp.isdir(osp.join(args.base_dir, d))]
    for conv_dir in (pbar_conv_dir := tqdm(conversion_dirs, position=0, leave=True)):
        pbar_conv_dir.set_description(f"Processing {osp.basename(osp.normpath(conv_dir))}")
        gender_seed_dirs = [osp.join(conv_dir, d) for d in os.listdir(conv_dir) if osp.isdir(osp.join(conv_dir, d))]
        for gender_seed_dir in (pbar_gender_seed := tqdm(gender_seed_dirs, position=1, leave=True)):
            pbar_gender_seed.set_description(f"Run {osp.basename(osp.normpath(gender_seed_dir))}")
            checkpoint_dir = osp.join(gender_seed_dir, 'checkpoints')
            checkpoint_files = [osp.join(checkpoint_dir, f) for f in os.listdir(checkpoint_dir) if osp.isfile(osp.join(checkpoint_dir, f)) and osp.splitext(osp.join(checkpoint_dir, f))[1] == '.ckpt']
            for ckpt_file in tqdm(checkpoint_files, position=2, leave=False):
                sanitized_fpath = sanitize_checkpoint_for_distribution(ckpt_file)
                sanitized_fname = osp.basename(sanitized_fpath)
                assert sanitized_fname in os.listdir(checkpoint_dir), f"Could not find sanitized checkpoint {sanitized_fname} in checkpoint directory"
