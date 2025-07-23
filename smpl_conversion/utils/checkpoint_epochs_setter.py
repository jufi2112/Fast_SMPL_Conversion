# Copyright (c) 2025 Julien Fischer, Chair of Computer Graphics and Visualization, TUD Dresden University of Technology.
# All rights reserved.
# Licensed under the MIT License. See LICENSE in the project root for license information.

import os
import torch
import argparse
from os import path as osp

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Changes the number of training epochs of the provided checkpoint")
    parser.add_argument("-c", "--ckpt", type=str, required=True,
                        help="Path to the checkpoint that should be modified")
    parser.add_argument('-e', '--epochs', type=int, required=True,
                        help="Number of epochs to which the checkpoint should be changed")
    args = parser.parse_args()
    if not osp.isfile(args.ckpt):
        raise ValueError(f"The provided checkpoint does not exist: {args.ckpt}")
    if not isinstance(args.epochs, int):
        raise ValueError(f"Epochs are required to be of type int, got {type(args.epochs)}")
    if args.epochs <= 0:
        raise ValueError(f"Epochs value must be greater zero, is {args.epochs}")
    ckpt = torch.load(args.ckpt)
    old_epochs = ckpt['train_progress']['config']['training']['epochs']
    ckpt['train_progress']['config']['training']['epochs'] = int(args.epochs)
    buffer = args.ckpt + '.tmp'
    torch.save(ckpt, buffer)
    os.replace(buffer, args.ckpt)
    print(f"Replaced old epoch value ({old_epochs}) with new epoch value ({int(args.epochs)}).")
    print(f"Checkpoint written.")
