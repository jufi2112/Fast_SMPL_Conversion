# Copyright (c) 2025 Julien Fischer, Chair of Computer Graphics and Visualization, TUD Dresden University of Technology.
# All rights reserved.
# Licensed under the MIT License. See LICENSE in the project root for license information.

import os
import yaml
import torch
import random
import pickle
import argparse
import numpy as np

from dotmap import DotMap
from os import path as osp
from time import perf_counter
from tqdm import tqdm as tqdm_shell
from torch.utils.data import DataLoader
from tqdm.notebook import tqdm as tqdm_notebook
from typing import Union, Dict, Callable, List, Tuple
from smpl_conversion.losses import CorrespondenceLoss
from smpl_conversion.data import SMPLConversionDataset
from smpl_conversion.utils.parameter_processing import direct_transfer_pose_parameters
from smpl_conversion.utils.enum_configurations import TqdmUsage, BodyModelType, PoseRepresentation

UNIT_FACTORS = {
    'm': 1,
    'cm': 100,
    'mm': 1000
}

class DirectParameterTransferEvaluator:
    def __init__(self, config: Union[Dict, str]):
        """
            Class that performs and evaluates direct parameter transfer between two body models.
            Translation and shape are transferred 'as-is', while pose parameters may need
            some form of adapation depending on the hierarchy.

        Params
        ------
            config (dict or str):
                Configuration file.
        """
        self.device = None
        self.config = None
        self.tqdm_usage = None
        self.target_body_type = self.input_body_type = None
        self.dataset = None
        self.data_loader = None
        self._load_config(config)
        self._set_device()
        self._process_data()


    def _load_config(self,
                     config: Union[str, Dict]
                     ):
        if isinstance(config, dict):
            self.config = config
        else:
            with open(config, 'r') as file:
                self.config = yaml.safe_load(file)
        self.config = DotMap(self.config, _dynamic=False)
        try:
            self.target_body_type = BodyModelType.from_string(self.config['conversion']['mode'].split('2')[1])
            self.input_body_type = BodyModelType.from_string(self.config['conversion']['mode'].split('2')[0])
        except KeyError:
            print("No conversion mode found, skipping this attribute...")
            pass
        if not 'tqdm' in self.config['general'].keys():
            self.tqdm_usage = TqdmUsage.OFF
        else:
            self.tqdm_usage = TqdmUsage.from_string(self.config['general']['tqdm'])
            if self.tqdm_usage is None:
                self.tqdm_usage = TqdmUsage.OFF
        self._set_random_seed()


    def _set_device(self):
        self.device = self.config.general.device
        if self.device == 'auto':
            self.device = "cuda" if torch.cuda.is_available() else "cpu"


    def _set_random_seed(self):
        np.random.seed(1)
        torch.manual_seed(1)
        random.seed(1)


    def _process_data(self):
        if self.tqdm_usage == TqdmUsage.SHELL:
            print("\n")
        print("Processing dataset...", end=' ')
        self.dataset = SMPLConversionDataset(
            self.config.data.dataset_path,
            self.config.conversion.mode,
            self.config.data.parameters_to_extract,
            self.config.conversion.parameter,
            self.config.general.gender,
            self.config.general.device_dataset,
            self.device,
            PoseRepresentation.ROTATION_VECTOR,
            None,   # use same convertor as above
            self.config.data.subsampling_factor,
            None, None,
            normalize_data=False
        )
        self.data_loader = DataLoader(self.dataset,
                                      batch_size=self.config.evaluation.batch_size)
        print("done.")


    def evaluate(self,
                 print_evaluation_results: bool = True
                 ) -> Tuple[Dict[str, float], float]:
        """
            Calculates metrics for direct parameter transfer on given data

            Params
            ------
                print_evaluation_results (bool):
                    Whether the evaluation results should be printed.
                    Defaults to True.

            Returns
            -------
                Dict[str, Dict[str, float]]:
                    Dict-like that contains metrics
                float:
                    Evaluation time in seconds.
        """
        loss_fn_metrics = CorrespondenceLoss(self.input_body_type,
                                             self.target_body_type,
                                             self.config.general.gender,
                                             self.config.general.body_model_location,
                                             self.config.general.transfer_file_location,
                                             self.config.evaluation.batch_size,
                                             self.config.data.n_shape_components,
                                             self.device,
                                             self.config.conversion.supr_is_constrained,
                                             PoseRepresentation.ROTATION_VECTOR,
                                             PoseRepresentation.ROTATION_VECTOR)
        metrics = self.config.evaluation.metrics

        if self.tqdm_usage == TqdmUsage.NOTEBOOK:
            loop = tqdm_notebook
        elif self.tqdm_usage == TqdmUsage.SHELL:
            loop = tqdm_shell
        
        num_metrics = len(metrics) * len(self.config.evaluation.metrics_reduction_mode)
        total_metrics = torch.zeros((num_metrics,), dtype=torch.float32)
        eval_time_start = perf_counter()

        for X, _ in self.data_loader if self.tqdm_usage == TqdmUsage.OFF else loop(self.data_loader, desc="Evaluating data", position=1, leave=True):
            metrics_nbr = self._evaluation_pass(X, metrics, loss_fn_metrics, len(self.dataset))
            total_metrics += metrics_nbr

        eval_time = perf_counter() - eval_time_start
        d = {}
        i = 0
        if self.tqdm_usage == TqdmUsage.SHELL and print_evaluation_results:
            print("\n")
        if print_evaluation_results:
            print(f"Evaluation time: {eval_time} seconds")
        for metric in self.config.evaluation.metrics:
            for red_mode in self.config.evaluation.metrics_reduction_mode:
                if print_evaluation_results:
                    print(f"{metric} {red_mode}: {total_metrics[i].item() * UNIT_FACTORS[args.unit]} {args.unit}")
                d[f'{metric}_{red_mode}'] = total_metrics[i].item()
                i += 1
        return d, eval_time


    def _evaluation_pass(self,
                         X: DotMap,
                         metrics: List[str],
                         loss_fn_metrics: Callable,
                         n_total_samples: int
                         ) -> torch.Tensor:
        num_metrics = len(metrics) * len(self.config.evaluation.metrics_reduction_mode)
        batch_metrics = torch.zeros((num_metrics,), dtype=torch.float32)
        bs = len(X.poses)
        batch_weight = bs / n_total_samples
        input_params = DotMap({
            'trans': X.trans,
            'betas': X.betas[..., :self.config.data.n_shape_components],
            'poses': X.poses
        }, _dynamic=False)
        target_params = DotMap({
            'trans': X.trans,
            'betas': X.betas[..., :self.config.data.n_shape_components],
            'poses': direct_transfer_pose_parameters(self.input_body_type,
                                                     self.target_body_type,
                                                     PoseRepresentation.ROTATION_VECTOR,
                                                     PoseRepresentation.ROTATION_VECTOR,
                                                     X.poses)
        }, _dynamic=False)
        i = 0
        with torch.no_grad():
            for metric in metrics:
                for red_mode in self.config.evaluation.metrics_reduction_mode:
                    batch_metrics[i] = loss_fn_metrics(input_params,
                                                       target_params,
                                                       loss_type=metric,
                                                       reduction_mode=red_mode,
                                                       param_names=self.config.conversion.parameter
                                                       ).item() * batch_weight
                    i += 1
        return batch_metrics


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Performs and evaluates direct parameter transfer.")
    parser.add_argument('-c', '--config', type=str, default=None, help="Path to the "
                        "configuration file.")
    parser.add_argument('-e', '--evaluation-dir', type=str, default=None, help="Path to "
                        "a previous evaluation run from which the results should be reported.")
    parser.add_argument('-u', '--unit', type=str, default='m',
                        help="Unit in which to show the errors. Defaults to meters (m)")
    args = parser.parse_args()
    if args.config is not None and args.evaluation_dir is not None:
        raise ValueError("Provide either --config or --evaluation-dir argument, not both!")
    if args.config is not None:
        if not osp.isfile(args.config):
            raise ValueError("Argument --config does not point to a valid file!")
        with open(args.config, 'r') as file:
            config = yaml.safe_load(file)
        genders = config['general']['gender']
        if isinstance(genders, str):
            genders = [genders]
        modes = config['conversion']['mode']
        if isinstance(modes, str):
            modes = [modes]
        output_location = config['evaluation']['output_location']
        if osp.splitext(output_location)[1] != '':
            output_location = osp.dirname(output_location)

        # tqdm settings
        if not 'tqdm' in config['general'].keys():
            tqdm_usage = TqdmUsage.OFF
        else:
            tqdm_usage = TqdmUsage.from_string(config['general']['tqdm'])
            if tqdm_usage is None:
                tqdm_usage = TqdmUsage.OFF
        if tqdm_usage == TqdmUsage.NOTEBOOK:
            loop = tqdm_notebook
        elif tqdm_usage == TqdmUsage.SHELL:
            loop = tqdm_shell

        datasets = config['data']['dataset_path']
        pbar_m = None
        for mode in modes if tqdm_usage == TqdmUsage.OFF else (pbar_m := loop(modes, leave=True, position=0)):
            if pbar_m is not None:
                pbar_m.set_description(f"Current mode: {mode}")
            config['conversion']['mode'] = mode
            conv_from, conv_to = mode.split('2')
            conv_from = BodyModelType.from_string(conv_from)
            conv_to = BodyModelType.from_string(conv_to)
            try:
                dataset_choice = datasets[conv_from.to_internal_string()]
            except KeyError:
                print(f"Did not find suitable dataset for mode {mode}, skipping...")
                continue
            if isinstance(dataset_choice, str):
                if osp.isdir(dataset_choice):
                    datasets_to_evaluate = [osp.join(dataset_choice, file) for file in os.listdir(dataset_choice) if osp.splitext(file)[1] == '.npz']
                else:
                    datasets_to_evaluate = [dataset_choice]
            else:
                datasets_to_evaluate = dataset_choice
            output_location_mode = osp.join(output_location, mode)
            os.makedirs(output_location_mode, exist_ok=True)
            pbar_d = None
            for dset in datasets_to_evaluate if tqdm_usage == TqdmUsage.OFF else (pbar_d := loop(datasets_to_evaluate, leave=True, position=1)):
                dset_name = osp.splitext(osp.basename(dset))[0]
                if pbar_d is not None:
                    pbar_d.set_description(f"Current dataset: {dset_name}")
                config['data']['dataset_path'] = dset
                pbar_g = None
                for gender in genders if tqdm_usage == TqdmUsage.OFF else (pbar_g := loop(genders, leave=True, position=2)):
                    if pbar_g is not None:
                        pbar_g.set_description(f"Current gender: {gender}")
                    output = osp.join(output_location_mode, 'direct_transfer_evaluation_' + dset_name + '_' + gender + '.pkl')
                    print(f"\nResults will be written to {output}")
                    config['evaluation']['output_location'] = output
                    config['general']['gender'] = gender
                    try:
                        evaluator = DirectParameterTransferEvaluator(config)
                        print(f"Mode: {mode}")
                        print(f"Dataset: {dset_name}")
                        print(f"Gender: {gender}")
                        d, runtime_s = evaluator.evaluate()
                    except Exception as e:
                        print(f"Encountered the following error during evaluation of dataset {dset_name}, skipping...")
                        print(repr(e))
                        continue
                    data = {
                        'dataset': dset_name,
                        'gender': gender,
                        'mode': mode,
                        'dataset_path': dset,
                        'evaluation_time_s': runtime_s,
                        'metrics': d
                    }
                    os.makedirs(osp.dirname(output), exist_ok=True)
                    with open(output, 'wb') as file:
                        pickle.dump(data, file, protocol=pickle.HIGHEST_PROTOCOL)
    elif args.evaluation_dir is not None:
        if not osp.isdir(args.evaluation_dir):
            raise ValueError("The provided --evaluation-dir argument is not a valid directory!")
        subdirs = [folder for folder in os.listdir(args.evaluation_dir) if osp.isdir(osp.join(args.evaluation_dir, folder))]
        reports = [osp.join(args.evaluation_dir, subdir, file) for subdir in subdirs for file in os.listdir(osp.join(args.evaluation_dir, subdir)) if osp.splitext(file)[1] == '.pkl']
        #[osp.join(args.evaluation_dir, file) for file in os.listdir(args.evaluation_dir) if osp.splitext(file)[1] == '.pkl']
        for report in reports:
            fname = osp.basename(report)
            try:
                with open(report, 'rb') as file:
                    r = pickle.load(file)
                dset_name = r['dataset']
                dset_path = r['dataset_path']
                gender = r.get('gender', 'Not specified')
                mode = r.get('mode', 'Not specified')
                evaluation_time = r['evaluation_time_s']
                evaluation_metrics = r['metrics']
                print(f'Performance report of direct parameter transfer for mode {mode} on dataset "{dset_name}" and gender {gender}')
                print(f"   Evaluation time: {evaluation_time} seconds")
                print(f"   Metrics:")
                for metric, val in evaluation_metrics.items():
                    print(f"      {metric}: {val * UNIT_FACTORS[args.unit]} {args.unit}")
                print("\n")
            except Exception as e:
                print(f"Encountered the following error while parsing {fname}:")
                print(repr(e))
                print("\n")
                continue
    else:
        raise ValueError("Need to provide either --config or --evaluation-dir argument")
