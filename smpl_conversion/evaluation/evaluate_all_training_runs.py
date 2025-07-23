# Copyright (c) 2025 Julien Fischer, Chair of Computer Graphics and Visualization, TUD Dresden University of Technology.
# All rights reserved.
# Licensed under the MIT License. See LICENSE in the project root for license information.

import os
import yaml
import vedo
import pickle
import argparse
import numpy as np

from os import path as osp
from tqdm import tqdm as tqdm_shell
from tqdm.notebook import tqdm as tqdm_notebook
from smpl_conversion.evaluation import CombinedEvaluator
from smpl_conversion.utils.enum_configurations import ExperimentType, BodyModelType
from smpl_conversion.utils.experiment_directory_manager import ExperimentDirectoryManager

UNIT_FACTORS = {
    'm': 1,
    'cm': 100,
    'mm': 1000
}

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Evaluates all training runs in a given directory. Expects directory structure to be e.g. base_dir/smplx2supr/male_seed-2")
    parser.add_argument('-b', '--base-dir', type=str, default=None, help="Path to the base directory that contains all training runs.")
    parser.add_argument('-o', '--output', type=str, required=True, help="Path to where the evaluation metrics should be written.")
    parser.add_argument('-c', '--config', type=str, default=None, help="Path to the evaluation configuration file.")
    parser.add_argument('-u', '--unit', type=str, default='m', help="Unit in which to show the errors. Defaults to meters (m)")
    parser.add_argument('--experiments', action='store_true', help="Whether multiple experiments are to be evaluated.")
    parser.add_argument('--skip-existing-data', action='store_true', help="If dataset already exists in evaluation log, skip without"
                        " re-evaluating it.")
    args = parser.parse_args()

    if args.base_dir is not None and args.config is not None:
        # Perform new evaluation run
        if not osp.isfile(args.config):
            raise ValueError(f"Provided config is not a valid file: {args.config}")
        if not osp.isdir(args.base_dir):
            raise ValueError(f"The provided base directory is not a valid directory: {args.base_dir}")

        os.makedirs(args.output, exist_ok=True)
        output_fpath = osp.join(args.output, 'evaluation_results')

        if osp.isfile(output_fpath+'.pkl'):
            with open(output_fpath+'.pkl', 'rb') as file:
                evaluation_results = pickle.load(file)
        else:
            evaluation_results = {}

        # Load config
        with open(args.config, 'r') as file:
            config = yaml.safe_load(file)
        if config['evaluation']['show_vertex_error_distribution']:
            vedo.settings.default_backend = 'vtk'
            vedo.settings.multi_samples = 8
        datasets = config['dataset']
        if isinstance(datasets, str):
            datasets = [datasets]

        if config['evaluation']['tqdm_mode'] == 'shell':
            loop = tqdm_shell
        elif config['evaluation']['tqdm_mode'] == 'notebook':
            loop = tqdm_notebook
        else:
            loop = None
        loop_pos = 0

        if args.experiments:
            experiments = [osp.join(args.base_dir, d) for d in os.listdir(args.base_dir) if osp.isdir(osp.join(args.base_dir, d))]
        else:
            experiments = [args.base_dir]

        for experiment in experiments if not args.experiments else loop(experiments, desc="Experiments", position=loop_pos, leave=True):
            if args.experiments:
                loop_pos += 1
            exp_name = osp.basename(experiment)
            if args.experiments:
                if exp_name not in evaluation_results.keys():
                    evaluation_results[exp_name] = {}
            # Search for all modes
            modes = [d for d in os.listdir(experiment) if osp.isdir(osp.join(experiment, d))]
            print(f"{f'For experiment {exp_name} f' if args.experiments else 'F'}ound the following modes for evaluation: {modes}")

            for mode in modes if loop is None else loop(modes, desc="Modes", position=loop_pos, leave=True):
                loop_pos += 1
                if args.experiments:
                    if mode not in evaluation_results[exp_name].keys():
                        evaluation_results[exp_name][mode] = {}
                else:
                    if mode not in evaluation_results.keys():
                        evaluation_results[mode] = {}
                mode_path = osp.join(experiment, mode)
                conv_from, conv_to = mode.split('2')
                conv_from = BodyModelType.from_string(conv_from)
                conv_to = BodyModelType.from_string(conv_to)
                # Search for all training runs in this mode
                training_runs = [d for d in os.listdir(mode_path) if osp.isdir(osp.join(mode_path, d))]

                for training_run in training_runs if loop is None else loop(training_runs, desc="Training runs", position=loop_pos, leave=True):
                    loop_pos += 1
                    training_run_path = osp.join(mode_path, training_run)
                    gender, seed = training_run.split('_')
                    seed = int(seed.split('-')[1])
                    if args.experiments:
                        if gender not in evaluation_results[exp_name][mode].keys():
                            evaluation_results[exp_name][mode][gender] = {}
                        if seed not in evaluation_results[exp_name][mode][gender].keys():
                            evaluation_results[exp_name][mode][gender][seed] = {}
                    else:
                        if gender not in evaluation_results[mode].keys():
                            evaluation_results[mode][gender] = {}
                        if seed not in evaluation_results[mode][gender].keys():
                            evaluation_results[mode][gender][seed] = {}

                    # Find latest checkpoint
                    ckpt_dir = osp.join(training_run_path, 'checkpoints')
                    files = [file for file in os.listdir(ckpt_dir) if osp.splitext(file)[1] in ['.ckpt']]
                    if len(files) == 0:
                        print(f"ERROR: Could not find checkpoint file for training run {mode}/{training_run}")
                        continue
                    # Sort by data modified
                    files.sort(key=lambda x: osp.getmtime(osp.join(ckpt_dir, x)))
                    ckpt_path = osp.join(ckpt_dir, files[-1])
                    config['checkpoint'] = ckpt_path

                    for dataset in datasets[conv_from.to_internal_string()] if loop is None else loop(datasets[conv_from.to_internal_string()], desc="Datasets", position=loop_pos, leave=True):
                        loop_pos += 1
                        config['dataset'] = dataset
                        dataset_name = osp.splitext(osp.basename(dataset))[0]
                        # Combined Evaluator will load existing evaluation results, so no skipping necessary
                        # this also allows to e.g. visualize results of a previous evaluation that were not visualized back then
                        if args.skip_existing_data:
                            if args.experiments:
                                if dataset_name in evaluation_results[exp_name][mode][gender][seed].keys():
                                    continue
                            else:
                                if dataset_name in evaluation_results[mode][gender][seed].keys():
                                    continue

                        # Start evaluation for this training run
                        edm = ExperimentDirectoryManager(base_dir=training_run_path,
                                                         experiment_type=ExperimentType.COMBINED_EVALUATION,
                                                         evaluation_config=config,
                                                         verbosity=config['verbosity'])
                        eval_config, _ = edm.get_modified_config_and_checkpoint()
                        comb_evaluator = CombinedEvaluator(eval_config['checkpoint'],
                                                           eval_config['dataset'],
                                                           eval_config['body_model_location'],
                                                           eval_config['transfer_file_location'],
                                                           eval_config['dataset_device'],
                                                           eval_config['inference_device'],
                                                           eval_config['verbosity']
                                                           )
                        result_statistics = comb_evaluator.evaluate(**(eval_config['evaluation']))
                        res_dict = {
                            'dataset_size': result_statistics['dataset_size'],
                            'inference_s': result_statistics['inference_s'],
                            'evaluation_s': result_statistics['evaluation_s'],
                            'batch_size': result_statistics['batch_size'],
                        }
                        if args.experiments:
                            evaluation_results[exp_name][mode][gender][seed][dataset_name] = res_dict
                            evaluation_results[exp_name][mode][gender][seed][dataset_name].update(result_statistics['metrics'])
                        else:
                            evaluation_results[mode][gender][seed][dataset_name] = res_dict
                            evaluation_results[mode][gender][seed][dataset_name].update(result_statistics['metrics'])

                        # Store evaluation results
                        with open(output_fpath+'.tmp', 'wb') as file:
                            pickle.dump(evaluation_results, file, pickle.HIGHEST_PROTOCOL)
                        os.replace(output_fpath+'.tmp', output_fpath+'.pkl')
                        del comb_evaluator
                        del edm
    else:
        # Load results of previous evaluation run
        if not osp.isfile(args.output):
            args.output = osp.join(args.output, 'evaluation_results.pkl')
            if not osp.isfile(args.output):
                raise ValueError("Neither --base-dir nor --config options are specified, so --output must point to a result file of a previous evaluation run")
        with open(args.output, 'rb') as file:
            evaluation_results = pickle.load(file)

    # Print evaluation run results
    if args.experiments:
        experiments = evaluation_results.keys()
    else:
        experiments = None

    for experiment in experiments if experiments is not None else [None]:
        for mode in evaluation_results.keys() if experiments is None else evaluation_results[experiment].keys():
            for gender in evaluation_results[mode].keys() if experiments is None else evaluation_results[experiment][mode].keys():
                metrics = {}
                for seed in evaluation_results[mode][gender].keys() if experiments is None else evaluation_results[experiment][mode][gender].keys():
                    for dataset in evaluation_results[mode][gender][seed].keys() if experiments is None else evaluation_results[experiment][mode][gender][seed].keys():
                        if dataset not in metrics.keys():
                            metrics[dataset] = {
                                'vertex': None,
                                'edge': None,
                                #'mse': None
                            }
                        stats = evaluation_results[mode][gender][seed][dataset] if experiments is None else evaluation_results[experiment][mode][gender][seed][dataset]
                        print(f"Results for {f'experiment {experiment} with ' if experiments is not None else ''}conversion {mode} with gender {gender} and seed {seed} on dataset {dataset}:")
                        print(f"   vertex: {stats['vertex'].item() * UNIT_FACTORS[args.unit]} {args.unit}")
                        print(f"   edge: {stats['edge'].item() * UNIT_FACTORS[args.unit]} {args.unit}")
                        #print(f"   mse: {stats['mse'].item()}")
                        metrics[dataset]['vertex'] = np.asarray(stats['vertex'].item(), dtype=np.float32) if metrics[dataset]['vertex'] is None else np.hstack([metrics[dataset]['vertex'], stats['vertex'].item()])
                        metrics[dataset]['edge'] = np.asarray(stats['edge'].item(), dtype=np.float32) if metrics[dataset]['edge'] is None else np.hstack([metrics[dataset]['edge'], stats['edge'].item()])
                        #metrics[dataset]['mse'] = np.asarray(stats['mse'].item(), dtype=np.float32) if metrics[dataset]['mse'] is None else np.hstack([metrics[dataset]['mse'], stats['mse'].item()])
                for dataset in metrics.keys():
                    print(f"Mean and standard deviation for conversion {mode} with gender {gender} on dataset {dataset}:")
                    for met in ['vertex', 'edge']:
                        print(f"   {met}:")
                        print(f"      mean: {np.mean(metrics[dataset][met]) * UNIT_FACTORS[args.unit]} {args.unit}")
                        print(f"      std dev: {np.std(metrics[dataset][met]) * UNIT_FACTORS[args.unit]} {args.unit}")
                        print(f"         values: {metrics[dataset][met]  * UNIT_FACTORS[args.unit]} {args.unit}")
