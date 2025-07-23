# Copyright (c) 2025 Julien Fischer, Chair of Computer Graphics and Visualization, TUD Dresden University of Technology.
# All rights reserved.
# Licensed under the MIT License. See LICENSE in the project root for license information.

import os
import yaml
import torch

from os import path as osp
from typing import Union, Dict, Tuple
from smpl_conversion.utils.enum_configurations import ExperimentType


class ExperimentDirectoryManager:
    def __init__(self,
                 base_dir: str,
                 experiment_type: Union[str, ExperimentType],
                 allow_checkpoint_loading: bool = True,
                 evaluation_config: Union[str, Dict] = None,
                 verbosity: int = 0):
        """
            Class that manages execution of different experiments in a specific
            base directory by modifying the configuration file.

        Params
        ------
            base_dir (str):
                Experiment base directory.
            experiment_type (str or ExperimentType):
                Type of the experiment that should be managed.
            allow_checkpoint_loading (bool):
                Whether training can continue from an existing checkpoint if
                a checkpoint was found. Optional, defaults to True
            evaluation_config (str or dict):
                If experiment type is evaluation, this is the path to the
                configuration file of the evaluation run, or a dictionary that
                contains the configuration. Defaults to None.
            verbosity (int):
                Verbosity level.
        """
        # Directory where checkpoints are written to. Should be "checkpoints" by default
        self.CHECKPOINT_DIR = "checkpoints"
        # Directory where training logs should be written to. Should be "train_logs" by default
        self.TRAIN_LOG_DIR = "train_logs"
        # Directory where evaluation results should be written to. Should be "evaluation" by default
        self.EVALUATION_DIR = "evaluation"

        if isinstance(experiment_type, ExperimentType):
            self.experiment_type = experiment_type
        else:
            self.experiment_type = ExperimentType.from_string(experiment_type)
            if self.experiment_type is None:
                raise ValueError(f"{experiment_type} is no valid ExperimentType!")
        self.evaluation_config = evaluation_config
        self.base_dir = base_dir
        self.verbosity = verbosity

        if self.experiment_type is None:
            raise ValueError(f"Unsupported experiment type: {experiment_type}")
        if self.verbosity > 1:
            print(f"Experiment type: {str(self.experiment_type)}")
        if not osp.isdir(base_dir):
            raise ValueError(f"Given base directory {base_dir} does not exist!")
        if self.verbosity > 1:
            print(f"Base directory: {base_dir}")
        if self.experiment_type == ExperimentType.COMBINED_EVALUATION and self.evaluation_config is None:
            raise ValueError(
                "If experiment type is combined evaluation, you have to provide the path to a configuration file "
                "or a dict containing the configuration via the 'evaluation_config' argument."
            )

        # Load configurationa and checkpoint
        config, self.ckpt = self._load_config_and_checkpoint(allow_checkpoint_loading)

        # Modify content of configuration
        self.config = self._modify_config(config, base_dir)

        if self.experiment_type == ExperimentType.COMBINED_TRAINING:
            if self.ckpt is not None:
                self.ckpt['train_progress']['config'] = self.config
        if self.experiment_type == ExperimentType.FINETUNING:
            if 'finetune_progress' not in self.ckpt:
                self.ckpt['finetune_progress'] = {}
            self.ckpt['finetune_progress']['config'] = self.config

        elif self.experiment_type == ExperimentType.COMBINED_EVALUATION:
            os.makedirs(self.config['evaluation']['result_save_location'], exist_ok=True)
            with open(osp.join(self.config['evaluation']['result_save_location'], 'evaluation_config.yaml'), "w+") as file:
                yaml.dump(self.config, file)
            if self.verbosity > 0:
                print(f"Created evaluation folder and wrote evaluation configuration to {self.config['evaluation']['result_save_location']}")


    def get_modified_config_and_checkpoint(self) -> Tuple[Dict, Union[Dict, None]]:
        """
            Returns the modified config file and, if available, the latest
            checkpoint with modified config included.

        Returns
        -------
            Dict:
                The configuration file modified in such a way that all output
                is written inside a base directory.
            None or Dict:
                If available, the loaded checkpoint with its configuration
                section being equivalent to the first return value (i.e. the
                modified configuration file), such that it can be directly
                used to continue training.
        """
        return self.config, self.ckpt


    def _load_config_and_checkpoint(self,
                                    allow_checkpoint_loading: bool
                                    ) -> Tuple[Union[Dict, None], Union[Dict, None]]:
        """
            Based on the experiment type, handles loading of configuration and checkpoint dictionaries.

        Params
        ------
            allow_checkpoint_loading (bool):
                Whether an existing checkpoint should be loaded (if supported by the experiment type).

        Returns
        -------
            Dict or None:
                Experiment configuration
            Dict or None:
                Experiment checkpoint
        """
        if self.experiment_type == ExperimentType.COMBINED_TRAINING:
            # Load Checkpoint if wanted
            if allow_checkpoint_loading:
                ckpt, ckpt_path = self._get_latest_checkpoint(osp.join(self.base_dir, self.CHECKPOINT_DIR))
            else:
                ckpt = None
                ckpt_path = None
            # Load configuration, either from checkpoint or from file
            if ckpt is not None:
                config = ckpt['train_progress']['config']
                if self.verbosity > 0:
                    print(f"Loaded configuration from checkpoint file at {ckpt_path}")
            else:
                config, config_path = self._get_config_file(self.base_dir)
                if config is None:
                    # Search in base dir
                    config, config_path = self._get_config_file(self.base_dir, prefer_config_folder=False)
                    if config is None:
                        raise ValueError(f"Could not find a configuration file inside {self.base_dir}")
                if self.verbosity > 0:
                    print(f"Loaded configuration from file {config_path}")
        elif self.experiment_type == ExperimentType.FINETUNING:
            # Load checkpoint
            ckpt, ckpt_path = self._get_latest_checkpoint(osp.join(self.base_dir, self.CHECKPOINT_DIR))
            if self.verbosity > 0:
                print(f"Found checkpoint at {ckpt_path}")
            if ckpt is None:
                raise ValueError(f"Could not find a checkpoint in directory {osp.join(self.base_dir, self.CHECKPOINT_DIR)}")
            # Load config file
            config, config_path = self._get_config_file(self.base_dir)
            if config is None:
                config, config_path = self._get_config_file(self.base_dir, prefer_config_folder=False)
                if config is None:
                    raise ValueError(f"Could not find a configuration .yaml file")
            if self.verbosity > 0:
                print(f"Found configuration file at {config_path}")
        elif self.experiment_type == ExperimentType.COMBINED_EVALUATION:
            ckpt = None
            if isinstance(self.evaluation_config, str):
                with open(self.evaluation_config, 'r') as file:
                    config = yaml.safe_load(file)
                    config_path = self.evaluation_config
                if self.verbosity > 0:
                    print(f"Loaded configuration from file {config_path}")
            elif isinstance(self.evaluation_config, dict):
                config = self.evaluation_config
                if self.verbosity > 0:
                    print("Copied configuration from provided config dictionary")
            else:
                raise ValueError(f"The provided evaluation_config argument is neither string nor dict, could not load combined evaluation configuration!")
        else:
            raise NotImplementedError(f"Loading configuration for experiment type {str(self.experiment_type)} is not implemented!")

        return config, ckpt



    def _get_config_file(self,
                         base_dir: str,
                         prefer_config_folder: bool = True
                         ) -> Tuple[Union[None, Dict], Union[None, str]]:
        """
            Searches for the experiment's configuration file and returns it.

        Params
        ------
            base_dir (str):
                Experiment base directory.
            prefer_config_folder (bool):
                If a diretory 'config' exists in the base_dir, search for
                configuration file inside this directory. Optional, defaults
                to True.

        Returns
        -------
            Dict or None:
                The configuration file if found, None if not found.
            str or None:
                The path of the configuration file, None if not found.
        """
        path_to_search = base_dir
        if osp.isdir(osp.join(base_dir, 'config')) and prefer_config_folder:
            path_to_search = osp.join(base_dir, 'config')
        files = os.listdir(path_to_search)
        yaml_files = [file for file in files if osp.splitext(file)[1] in ['.yml', '.yaml']]
        if len(yaml_files) == 0:
            return None, None
        if len(yaml_files) > 1:
            raise ValueError(f"Found multiple configuration files in {path_to_search}")
        path_config = osp.join(path_to_search, yaml_files[0])
        with open(path_config, 'r') as file:
            config = yaml.safe_load(file)
        return config, path_config


    def _modify_config(self,
                       config: Dict,
                       base_dir: str
                       ) -> Dict:
        """
            Modifies the given configuration such that its output is only in a
            specified directory.

        Params
        ------
            config (Dict):
                The configuration file that should be modified
            base_dir (str):
                Directory where all output should be placed in.
        """
        if self.experiment_type in [ExperimentType.COMBINED_TRAINING, ExperimentType.FINETUNING]:
            ckpt_dir = osp.join(base_dir, self.CHECKPOINT_DIR)
            train_log_dir = osp.join(base_dir, self.TRAIN_LOG_DIR)

            os.makedirs(ckpt_dir, exist_ok=True)
            os.makedirs(train_log_dir, exist_ok=True)

            config['training']['checkpoint_dir'] = ckpt_dir
            config['training']['log_dir'] = train_log_dir

            if self.verbosity > 1:
                print(f"Setting checkpoint directory to {ckpt_dir}")
                print(f"Setting train log directory to {train_log_dir}")
            if self.experiment_type == ExperimentType.FINETUNING:
                ckpt_config = self.ckpt['train_progress']['config']
                # Make sure that checkpoint and config aim for same conversion mode
                assert ckpt_config['conversion']['mode'] == config['conversion']['mode'], f"Checkpoint and config have different conversion modes: {ckpt_config['conversion']['mode']} vs {config['conversion']['mode']}"
                # Copy architecture details from checkpoint to config
                config['conversion'] = ckpt_config['conversion']
                config['transformer_settings'] = ckpt_config.get('transformer_settings', None)
                config['data']['n_shape_components'] = ckpt_config['data']['n_shape_components']
                config['data']['parameters_to_extract'] = ckpt_config['conversion']['parameter']
                if self.verbosity > 1:
                    print("Copied architecture settings from checkpoint to config")

        elif self.experiment_type == ExperimentType.COMBINED_EVALUATION:
            dataset_name = osp.splitext(osp.basename(config['dataset']))[0]
            result_save_location = osp.join(base_dir, self.EVALUATION_DIR, dataset_name)
            config['evaluation']['result_save_location'] = result_save_location
            _, ckpt_path = self._get_latest_checkpoint(osp.join(base_dir, self.CHECKPOINT_DIR),
                                                       load_checkpoint=False)
            config['checkpoint'] = ckpt_path

            if self.verbosity > 1:
                print(f"Setting output directory to {result_save_location}")
                print(f"Setting checkpoint file to {ckpt_path}")

        else:
            raise NotImplementedError(f"The experiment type {str(self.experiment_type)} is not supported.")
        return config


    def _get_latest_checkpoint(self,
                               checkpoint_dir: str,
                               load_checkpoint: bool = True
                               ) -> Tuple[Union[Dict, None], Union[str, None]]:
        """
            Returns the latest checkpoint (by date modified) if it exists,
            otherwise None.

        Params
        ------
            checkpoint_dir (str):
                Directory where to search for checkpoints.
            load_checkpoint (bool):
                Wehther the checkpoint should be loaded. If False, None is
                returned as first return parameter. Defaults to True.

        Returns
        -------
            Dict or None:
                The loaded latest checkpoint or None if no checkpoint
                was found.
            str or None:
                The path of the loaded checkpoint or None
        """
        if not osp.isdir(checkpoint_dir):
            return None, None
        files = os.listdir(checkpoint_dir)
        files = [file for file in files if osp.splitext(file)[1] in ['.ckpt']]
        if len(files) == 0:
            return None, None
        # Sort by date modified
        files.sort(key=lambda x: osp.getmtime(osp.join(checkpoint_dir, x)))

        # Load Checkpoint and return it
        path_ckpt = osp.join(checkpoint_dir, files[-1])
        if load_checkpoint:
            ckpt = torch.load(path_ckpt)
        else:
            ckpt = None
        return ckpt, path_ckpt
