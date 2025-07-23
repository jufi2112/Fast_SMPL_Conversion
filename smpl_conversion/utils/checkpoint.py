# Copyright (c) 2025 Julien Fischer, Chair of Computer Graphics and Visualization, TUD Dresden University of Technology.
# All rights reserved.
# Licensed under the MIT License. See LICENSE in the project root for license information.

import torch
import argparse
from copy import deepcopy
from dotmap import DotMap
from os import path as osp
from typing import Union, Dict, Optional
from smpl_conversion.utils.enum_configurations import ConversionNetworkArchitecture

def sanitize_checkpoint_for_distribution(ckpt: Union[str, Dict],
                                         output_fpath: Optional[str] = None
                                         ) -> str:
    """
        Removes all entries from the checkpoint that are not absolutely
        necessary to use it for predictions.
    
    Params
    ------
        ckpt (str or Dict):
            Either a filepath to the checkpoint or the already loaded checkpoint
            which should be sanitized.
        output_fpath (str):
            The file path (including filename) to which the sanitized checkpoint
            should be written to. If ckpt is a string, the file path and file
            name will be inferred from it and this parameter is not used.

    Returns
    -------
        str: The location to which the sanitized checkpoint has been written.
    """
    BLACKLIST = ['train_progress', 'optimizer_state_dict',
                 'lr_scheduler_state_dict', 'torch_random_state',
                 'numpy_random_state', 'numpy_rng_bit_generator_state']
    WHITELIST = ['gender', 'mode', 'parameter', 'model_max_dims',
                 'network_norm_layer', 'model_type', 'n_shape_components',
                 'normalize_data', 'normalization_method', 'supr_is_constrained',
                 'input_rotation_representation', 'output_rotation_representation',
                 'network_activation', 'normalization_methods', ]
    sanitized_ckpt = {}
    if not isinstance(ckpt, str) and not isinstance(ckpt, dict):
        raise ValueError("Argument 'ckpt' has to be either a string or a dict-like")
    if isinstance(ckpt, str):
        fpath = osp.dirname(ckpt)
        old_fname, ext = osp.splitext(osp.basename(ckpt))
        fname = old_fname + "_sanitized" + ext
        output_fpath = osp.join(fpath, fname)
        ckpt = torch.load(ckpt)
    else:
        if output_fpath is None:
            raise ValueError(
                "If argument 'ckpt' is a dict-like, you have to also provide "
                "the argument 'output_fpath'"
            )
    # Extract information necessary to build the network architecture
    if 'train_progress' in ckpt.keys():
        config = DotMap(_dynamic=False)
        ckpt_config = ckpt['train_progress']['config']
        for category, entry in ckpt_config.items():
            if isinstance(entry, dict):
                if category == 'transformer_settings':
                    ckpt_mode = ConversionNetworkArchitecture.from_string(ckpt_config['conversion']['mode'])
                    if ckpt_mode in [ConversionNetworkArchitecture.SEPARATED_ATTENTION,
                                     ConversionNetworkArchitecture.SEPARATED_ATTENTION_SKIP]:
                        config[category] = entry
                else:
                    for key, value in entry.items():
                        if key in WHITELIST:
                            if category in config.keys():
                                config[category][key] = value
                            else:
                                config[category] = DotMap({key: value}, _dynamic=False)
            else:
                if category in WHITELIST:
                    config[category] = entry
        sanitized_ckpt['train_progress'] = DotMap({'config': config}, _dynamic=False)
    
    for key, value in ckpt.items():
        if key not in BLACKLIST:
            sanitized_ckpt[key] = deepcopy(value)
    sanitized_ckpt['is_sanitized'] = True

    torch.save(sanitized_ckpt, output_fpath)
    return output_fpath

    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Sanitizes a checkpoint for distribution")
    parser.add_argument('file')
    args = parser.parse_args()
    loc = sanitize_checkpoint_for_distribution(args.file)
    print(f"Sanitized checkpoint has been written to {loc}")
