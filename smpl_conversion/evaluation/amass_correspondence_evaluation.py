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
from tqdm import tqdm as tqdm_shell
from torch.utils.data import DataLoader
from tqdm.notebook import tqdm as tqdm_notebook
from typing import Union, Dict, Callable, List, Tuple
from smpl_conversion.losses import CorrespondenceLoss
from smpl_conversion.data import SMPLConversionDataset
from smpl_conversion.utils.enum_configurations import TqdmUsage, BodyModelType, PoseRepresentation


class AMASSCorrespondenceEvaluator:
    def __init__(self, config: Union[Dict, str]):
        """
            Class that evaluates the correspondence quality between samples
            from the AMASS dataset.

            Params
            ------
                config (dict or str):
                    Configuration file
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
                     config: Union[str, Dict]):
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
            print("No valid conversion mode found, skipping...")
        if not 'tqdm' in self.config['general'].keys():
            self.tqdm_usage = TqdmUsage.OFF
        else:
            self.tqdm_usage = TqdmUsage.from_string(self.config['general']['tqdm'])
            if self.tqdm_usage is None:
                self.tqdm_usage = TqdmUsage.OFF
        self._set_random_seed()


    def _set_random_seed(self):
        np.random.seed(1)
        torch.manual_seed(1)
        random.seed(1)


    def _set_device(self):
        self.device = self.config.general.device
        if self.device == 'auto':
            self.device = "cuda" if torch.cuda.is_available() else "cpu"


    def _process_data(self):
        if self.tqdm_usage == TqdmUsage.SHELL:
            print("\n")
        print("Processing dataset...")
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
            None, None, False
        )
        self.data_loader = DataLoader(self.dataset,
                                      batch_size=self.config.evaluation.batch_size)
        print("done.")


    def evaluate(self,
                 print_evaluation_results: bool = True
                 ) -> Dict[str, float]:
        """
            Calculates metrics for AMASS correspondences on the given data

            Params
            ------
                print_evaluation_results (bool):
                    Whether the obtained metrics should be printed. Defaults to True

            Returns
            -------
                Dict[str, float]:
                    Evaluation metrics
        """
        metrics_fn = CorrespondenceLoss(self.input_body_type,
                                        self.target_body_type,
                                        self.config.general.gender,
                                        self.config.general.body_model_location,
                                        self.config.general.transfer_file_location,
                                        self.config.evaluation.batch_size,
                                        self.config.data.n_shape_components,
                                        self.device,
                                        False,
                                        PoseRepresentation.ROTATION_VECTOR,
                                        PoseRepresentation.ROTATION_VECTOR)
        metrics = self.config.evaluation.metrics
        if self.tqdm_usage == TqdmUsage.NOTEBOOK:
            loop = tqdm_notebook
        elif self.tqdm_usage == TqdmUsage.SHELL:
            loop = tqdm_shell

        num_metrics = len(metrics) * len(self.config.evaluation.metrics_reduction_mode)
        total_metrics = torch.zeros((num_metrics,), dtype=torch.float32)

        for X, y in self.data_loader if self.tqdm_usage == TqdmUsage.OFF else loop(self.data_loader, desc="Evaluating data", position=3, leave=True):
            metrics_nbr = self._evaluation_pass(X, y, metrics, metrics_fn, len(self.dataset))
            total_metrics += metrics_nbr

        d = {}
        i = 0
        dset_name = osp.splitext(osp.basename(self.config.data.dataset_path))[0]
        if self.tqdm_usage == TqdmUsage.SHELL and print_evaluation_results:
            print("\n")
        if print_evaluation_results:
            print(f"Evaluation results for mode {self.config.conversion.mode} with gender {self.config.general.gender} on dataset {dset_name}:")
        for metric in self.config.evaluation.metrics:
            for red_mode in self.config.evaluation.metrics_reduction_mode:
                if print_evaluation_results:
                    print(f"   {metric} {red_mode}: {total_metrics[i].item()}")
                d[f'{metric}_{red_mode}'] = total_metrics[i].item()
                i += 1
        return d


    def _evaluation_pass(self,
                         X: DotMap,
                         y: DotMap,
                         metrics: List[str],
                         metrics_fn: Callable,
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
            'trans': y.trans,
            'betas': y.betas[..., :self.config.data.n_shape_components],
            'poses': y.poses
        }, _dynamic=False)
        i = 0
        with torch.no_grad():
            for metric in metrics:
                for red_mode in self.config.evaluation.metrics_reduction_mode:
                    batch_metrics[i] = metrics_fn(input_params,
                                                  target_params,
                                                  loss_type=metric,
                                                  reduction_mode=red_mode,
                                                  param_names=self.config.conversion.parameter
                                                  ).item() * batch_weight
                    i += 1
        return batch_metrics

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Evaluates AMASS correspondences")
    parser.add_argument('-c', '--config', type=str, default=None,
                        help="Path to the configuration file")
    parser.add_argument('-r', '--result-file', type=str, default=None,
                        help="Path to a result file who's results should be "
                        "printed.")
    args = parser.parse_args()
    if args.config is None and args.result_file is None:
        raise ValueError("Either --config or --result-file argument must be provided!")
    if args.config is not None:
        if not osp.isfile(args.config):
            raise ValueError("The provided configuration file does not exist!")
        with open(args.config, 'r') as file:
            config = yaml.safe_load(file)
        modes = config['conversion']['mode']
        genders = config['general']['gender']
        datasets = config['data']['dataset_path']
        if isinstance(datasets, str):
            if osp.isdir(datasets):
                datasets = [osp.join(datasets, file) for file in os.listdir(datasets) if osp.splitext(file)[1] == '.npz']
            else:
                datasets = [datasets]
        if not osp.isdir(config['evaluation']['output_location']):
            os.makedirs(config['evaluation']['output_location'])
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

        results = {}
        for mode in modes if tqdm_usage == TqdmUsage.OFF else (pbar_m := loop(modes, leave=True, position=0)):
            pbar_m.set_description(f"Current mode: {mode}")
            config['conversion']['mode'] = mode
            results[mode] = {}
            for dset in datasets if tqdm_usage == TqdmUsage.OFF else (pbar_d := loop(datasets, leave=True, position=1)):
                dset_name = osp.splitext(osp.basename(dset))[0]
                pbar_d.set_description(f"Current dataset: {dset_name}")
                config['data']['dataset_path'] = dset
                results[mode][dset_name] = {}
                for gender in genders if tqdm_usage == TqdmUsage.OFF else (pbar_g := loop(genders, leave=True, position=2)):
                    pbar_g.set_description(f"Current gender: {gender}")
                    config['general']['gender'] = gender

                    evaluator = AMASSCorrespondenceEvaluator(config)
                    metrics = evaluator.evaluate()
                    results[mode][dset_name][gender] = metrics
                    del evaluator
        
        with open(osp.join(config['evaluation']['output_location'], 'results.pkl'), 'wb') as file:
            pickle.dump(results, file, protocol=pickle.HIGHEST_PROTOCOL)
    else:
        if not osp.isfile(args.result_file):
            raise ValueError("The provided result file does not exist!")
        with open(args.result_file, 'rb') as file:
            results = pickle.load(file)
        for mode in results.keys():
            for dataset in results[mode].keys():
                for gender in results[mode][dataset].keys():
                    metrics = results[mode][dataset][gender]
                    print(f"AMASS correspondences for mode {mode} with gender {gender} on dataset {dataset}:")
                    for metric_name, metric_val in metrics.items():
                        print(f"   {metric_name}: {metric_val}")
