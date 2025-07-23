# Copyright (c) 2025 Julien Fischer, Chair of Computer Graphics and Visualization, TUD Dresden University of Technology.
# All rights reserved.
# Licensed under the MIT License. See LICENSE in the project root for license information.

import os
import yaml
import math
import torch
import pickle
import random
import numpy as np

from dotmap import DotMap
from os import path as osp
from abc import ABC, abstractmethod
from torch.utils.data import DataLoader
from typing import Tuple, Dict, Union, Callable, List
from smpl_conversion.data import SMPLConversionDataset
from torch.utils.data.sampler import SubsetRandomSampler
from smpl_conversion.factories.optimizer import OptimizerFactory
from smpl_conversion.factories.lr_scheduler import LRSchedulerFactory
from smpl_conversion.utils.enum_configurations import TqdmUsage, ShapeUsage, PoseRepresentation, BodyModelType



class BaseTrainer(ABC):
    def __init__(self,
                 config: Union[str, Dict],
                 **kwargs):
        """
            Base class for all Trainer classes.

            Params
            ------
                config (str or Dict)
                    Either a path to a configuration file or an already loaded configuration.
        """
        self.config = None
        self.target_body_type = None
        self.input_body_type = None
        self.train_dataset = None
        self.model = None
        self.tqdm_usage = None
        self.shape_usage = None
        self.train_indices = None
        self.valid_indices = None
        self.normalizers = None
        self.input_rotation_representation = None
        self.output_rotation_representation = None
        self.evaluate_train_data_after_last_batch = False
        self.overall_train_data_indices = None
        self.check_for_nan = False
        self._load_config(config)
        #self.config.update(**kwargs)
        self._process_data()


    @abstractmethod
    def _set_random_seed(self):
        pass

    @abstractmethod
    def _process_data(self):
        pass

    @abstractmethod
    def fit(self):
        pass

    @abstractmethod
    def predict(self):
        pass


    def _load_config(self,
                     config: Union[str, Dict]):
        """
            Loads the provided configuration and parses it into the correct format.

            Params
            ------
                config (str or dict)
                    Either a path to a configuration file or an already loaded configuration.
        """
        if isinstance(config, dict):
            self.config = config
        else:
            with open(config, 'r') as file:
                self.config = yaml.safe_load(file)
        self.config = DotMap(self.config, _dynamic=False)
        # Check whether rotation representation fields are present.
        # If not, set them to default value
        self.input_rotation_representation = PoseRepresentation.from_string(self.config['conversion'].get('input_rotation_representation', None))
        if self.input_rotation_representation is None:
            print("No or invalid entry for config field 'conversion.input_rotation_representation', defaulting to rotation vector representation!")
            self.input_rotation_representation = PoseRepresentation.ROTATION_VECTOR
            self.config['conversion']['input_rotation_representation'] = 'rotation_vector'
        self.output_rotation_representation = PoseRepresentation.from_string(self.config['conversion'].get('output_rotation_representation', None))
        if self.output_rotation_representation is None:
            print("No or invalid entry for config field 'conversion.output_rotation_representation', defaulting to rotation vector representation")
            self.output_rotation_representation = PoseRepresentation.ROTATION_VECTOR
            self.config['conversion']['output_rotation_representation'] = 'rotation_vector'
        try:
            self.target_body_type = BodyModelType.from_string(self.config['conversion']['mode'].split('2')[1])
            self.input_body_type = BodyModelType.from_string(self.config['conversion']['mode'].split('2')[0])
        except KeyError:
            print("No conversion mode found, skipping this attribute...")
            pass
        if self.target_body_type is None:
            print("Invalid target body model type")
        if self.input_body_type is None:
            print("Invalid input body model type")
        if not 'tqdm' in self.config['general'].keys():
            self.tqdm_usage = TqdmUsage.OFF
        else:
            self.tqdm_usage = TqdmUsage.from_string(self.config['general']['tqdm'])
            if self.tqdm_usage is None:
                self.tqdm_usage = TqdmUsage.OFF
        if not 'shape_mode' in self.config['general'].keys():
            self.shape_usage = ShapeUsage.LEARN
        else:
            self.shape_usage = ShapeUsage.from_string(self.config['general']['shape_mode'])
            if self.shape_usage is None:
                print(
                    f"Invalid entry for general.shape_mode: {self.config['general']['shape_mode']}\n"
                    "Defaulting to learning shape parameters."
                )
                self.shape_usage = ShapeUsage.LEARN
        self.evaluate_train_data_after_last_batch = self.config['training'].get('evaluate_train_data_after_last_batch', False)
        self.check_for_nan = self.config['training'].get('check_for_nan', False)
        self._set_random_seed()



class AbstractNNTrainer(BaseTrainer):
    """
        Base class for all neural network-based trainers.

    Params
    ------
        config (str or dict):
            Either a path to a configuration file or an already loaded configuration.
    """
    def __init__(self,
                 config: Union[str, Dict],
                 **kwargs):
        self.train_loader = None
        self.valid_loader = None
        self.optimizer = None
        self.lr_scheduler = None
        self.device = None
        self.np_rng = None
        super().__init__(config, **kwargs)


    @abstractmethod
    def _create_model(self):
        pass


    def _load_config(self,
                     config: Union[str, Dict]):
        super()._load_config(config)
        self._set_device()


    def _set_random_seed(self):
        np.random.seed(self.config['general']['random_seed'])
        torch.manual_seed(self.config['general']['random_seed'])
        self.np_rng = np.random.default_rng(seed=self.config['general']['random_seed'])
        random.seed(self.config['general']['random_seed'])


    def _set_device(self):
        self.device = self.config['general']['device']
        if self.device == 'auto':
            self.device = "cuda" if torch.cuda.is_available() else "cpu"


    def _save_model_checkpoint(self,
                               current_loss,
                               lowest_loss,
                               current_epoch,
                               train_progress,
                               fname,
                               **kwargs):
        """
            Determines whether a checkpoint should be created based on the checkpoint
            settings in the configuration. Therefore, this function can be called in each epoch.
            Checkpointing is done by writing to a temporary file followed by an atomic renaming
            operation, such that the final .ckpt file should always be valid.

            Params
            ------
                current_loss (float): Loss of this epoch. Is used for checkpoint mode 'best'.
                lowest_loss (float): Lowest currently observed loss. Is used for checkpoint mode 'best'.
                current_epoch (int): Current training epoch. Used for checkpoint mode 'all'.
                train_progress (dict): Training progress that will be written to the checkpoint.
                fname (str): Name of the checkpoint file.
        """
        ckpt_mode = self.config['training']['save_checkpoint']
        if ckpt_mode == 'off':
            return
        if ckpt_mode == 'best' and current_loss >= lowest_loss:
            return
        if ckpt_mode == 'all':
            fname += f'_epoch-{current_epoch}'
        ckpt = {
            'finetune_progress' if self.finetune else 'train_progress': train_progress,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'lr_scheduler_state_dict': self.lr_scheduler.state_dict() if self.lr_scheduler else None,
            'torch_random_state': torch.get_rng_state(),
            'numpy_random_state': np.random.get_state(),
            'numpy_rng_bit_generator_state': self.np_rng.bit_generator.state,
            'normalizers': self.normalizers,
            'model_architecture': str(self.model)
        }
        ckpt.update(**kwargs)
        torch.save(
            ckpt,
            osp.join(
                self.config['training']['checkpoint_dir'],
                f"{fname}.tmp"
            )
        )
        os.replace(
            osp.join(
                self.config['training']['checkpoint_dir'],
                f"{fname}.tmp"
            ),
            osp.join(
                self.config['training']['checkpoint_dir'],
                f"{fname}.ckpt"
            )
        )


    def _load_checkpoint(self,
                         checkpoint_file: str) -> Tuple[str, float, float, Dict, Dict]:
        """
            Initializes the configuration, model, optimizer, and learning rate
            scheduler from the provided checkpoint. Also performs data preparation
            according to the configuration provided in the checkpoint.
            After this function, the class should be in the same state as when
            the checkpoint was created.

            Params
            ------
                checkpoint_file (str):
                    Path to the checkpoint file which should be loaded.

            Note
            ----
                This function is not used when creating a combined trainer class from a checkpoint.
        """
        print("Loading checkpoint...")
        ckpt = torch.load(checkpoint_file)
        self._load_config(ckpt['train_progress']['config'])
        self._process_data()
        self._create_model()
        self.model.load_state_dict(ckpt['model_state_dict'])
        last_epoch = sorted(ckpt['train_progress']['epochs'].keys())[-1]
        self._create_optimizer_and_lr_scheduler()
        self.optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        if self.lr_scheduler:
            self.lr_scheduler.load_state_dict(ckpt['lr_scheduler_state_dict'])
        self.normalizers = ckpt.get('normalizers', None)
        torch.set_rng_state(ckpt['torch_random_state'])
        np.random.set_state(ckpt['numpy_random_state'])
        if 'numpy_rng_bit_generator_state' in ckpt.keys():
            self.np_rng.bit_generator.state = ckpt['numpy_rng_bit_generator_state']
        lowest_valid_loss = np.min(
            np.asarray(
                [
                    ckpt['train_progress']['epochs'][epoch]['validation']['loss']
                    for epoch in sorted(ckpt['train_progress']['epochs'].keys())
                ]
            )
        )
        training_time = ckpt['train_progress']['epochs'][last_epoch]['training_time_s']
        # Return other top-level entries to let the calling function handle them
        other_entries = {}
        for key, val in ckpt.items():
            if key in ['train_progress', 'model_state_dict', 'optimizer_state_dict', 'lr_scheduler_state_dict', 'torch_random_state', 'numpy_random_state', 'normalizers', 'model_architecture']:
                continue
            other_entries[key] = val
        print("Checkpoint loading done.")
        return (
            ckpt['train_progress']['start'], training_time, lowest_valid_loss,
            ckpt['train_progress'], other_entries
        )


    def _create_optimizer_and_lr_scheduler(self,
                                           optim_args_overwrite=None,
                                           lr_scheduler_args_overwrite=None):
        """
            Creates the optimizer and learning rate scheduler provided by the configuration file.
            Optionally, the configuration file arguments can be overwritten through the function parameters.

            Params
            ------
                optim_args_overwrite (dict):
                    Optional, dict that contains the kwargs of the optimizer that should be overwritten.
                    Defaults to None (no overwriting).
                lr_scheduler_args_overwrite (dict):
                    Optional, dict that contains the kwargs of the learning rate
                    scheduler that should be overwritten. Defaults to None (no overwriting).
        """
        if not optim_args_overwrite:
            optim_args_overwrite={}
        if not lr_scheduler_args_overwrite:
            lr_scheduler_args_overwrite={}

        optim_args = self.config.training.optim_args
        if optim_args is None:
            optim_args = {}
        else:
            optim_args = optim_args.get(self.config.training.optimizer, None)
            if optim_args is None:
                optim_args = {}
                print(
                    f"Could not find an entry for optimizer {self.config.training.optimizer} "
                    "in configuration file under training.optim_args, assuming "
                    " all default arguments"
                )
            else:
                optim_args = dict(optim_args)
        optim_args['params'] = self.model.parameters()
        optim_args['lr'] = self.config.training.lr
        optim_args.update(optim_args_overwrite)

        self.optimizer = OptimizerFactory.create_optimizer(
            optim_name=self.config.training.optimizer,
            optim_args=optim_args
        )

        use_scheduling = True
        scheduler_name = ''
        if 'use_lr_scheduling' in self.config.training.keys():
            use_scheduling = self.config.training.use_lr_scheduling
            scheduler_name = 'reduce_lr_on_plateau'
        else:
            if self.config.training.lr_scheduler is None:
                use_scheduling = False
            else:
                scheduler_name = self.config.training.lr_scheduler

        if use_scheduling:
            lr_scheduler_args = self.config.training.lr_scheduler_args
            if lr_scheduler_args is None:
                lr_scheduler_args = {}
            else:
                lr_scheduler_args = lr_scheduler_args.get(self.config.training.lr_scheduler, None)
                if lr_scheduler_args is None:
                    lr_scheduler_args = {}
                    print(
                        f"Could not find an entry for scheduler {scheduler_name} "
                        "in configuration file under training.lr_scheduler_args, "
                        "assuming all default arguments."
                    )
                else:
                    lr_scheduler_args = dict(lr_scheduler_args)
            lr_scheduler_args['optimizer'] = self.optimizer
            lr_scheduler_args.update(lr_scheduler_args_overwrite)
            for k, v in lr_scheduler_args.items():
                if v == 'n_epochs':
                    lr_scheduler_args[k] = self.config.training.epochs
            self.lr_scheduler = LRSchedulerFactory.create_lr_scheduler(
                scheduler_name=scheduler_name,
                scheduler_args=lr_scheduler_args
            )
        return


    def _process_data(self):
        """
            Creates a dataset object with the parameters provided in the configuration file.
        """
        print("Processing dataset...", end=' ')
        self.train_dataset = SMPLConversionDataset(
            self.config.data.train_dataset_path if "train_dataset_path" in self.config.data.keys() else self.config.data.dataset_path,
            self.config.conversion.mode,
            self.config.data.parameters_to_extract,
            self.config.conversion.parameter,
            self.config.general.gender,
            self.config.general.device_dataset,
            self.device,
            self.input_rotation_representation,
            self.output_rotation_representation,
            1,
            None,
            None,
            normalize_data=self.config.data.normalize_data and not self.config.data.use_random_training_data,
            normalization_method=self.config.data.normalization_method,
            normalization_parameters=self.config.data.normalization_methods[self.config.data.normalization_method].kwargs,
            allow_unsupervised=self.config.data.allow_unsupervised
        )
        self.overall_train_data_indices = list(range(len(self.train_dataset)))
        if self.config['data']['shuffle_dataset']:
            self.np_rng.shuffle(self.overall_train_data_indices)
        if self.config.data.get("valid_dataset_path", None) is not None:
            self.valid_dataset = SMPLConversionDataset(
                self.config.data.valid_dataset_path,
                self.config.conversion.mode,
                self.config.data.parameters_to_extract,
                self.config.conversion.parameter,
                self.config.general.gender,
                self.config.general.device_dataset,
                self.device,
                self.input_rotation_representation,
                self.output_rotation_representation,
                self.config.data.subsampling_factor_valid,
                None,
                None,
                normalize_data=self.config.data.normalize_data and not self.config.data.use_random_training_data,
                normalization_method=self.config.data.normalization_method,
                normalization_parameters=self.config.data.normalization_methods[self.config.data.normalization_method].kwargs,
                allow_unsupervised=self.config.data.allow_unsupervised
            )
        else:
            self.valid_dataset = None
        # Data loaders are created at the beginning of each epoch
        print("done.")


    def _create_data_loaders(self):
        """
            Creates data loaders for the previously created dataset. Raises an error if no dataset has been
            created yet.
        """
        raise NotImplementedError("_create_data_loader is deprecated, use _recreate_random_data_loaders instead")


    def _recreate_random_data_loaders(self,
                                      subsampling_factor: float,
                                      current_epoch: int):
        """
            Recreates the random data loaders such that a new
            training dataset, sampled from the original training
            dataset with the provided subsampling factor, is available.
            Should be called at the beginning of each epoch.

            Params
            ------
                subsampling_factor (float):
                    Subsampling factor with which the original
                    training dataset should be sampled.
                current_epoch (int):
                    Number of the current epoch (0-based)
        """
        if self.train_dataset is None:
            raise ValueError("No training dataset loaded.")
        training_elements_per_epoch = math.floor(len(self.overall_train_data_indices) / subsampling_factor)
        # based on the current epoch, calculate which batch of training data should be used
        training_data_current_batch_idx = current_epoch % subsampling_factor
        if self.valid_dataset is None:
            # Use subset of training data for validation
            # all data up to this point will be in the training dataset for the current epoch
            data_this_epoch = self.overall_train_data_indices[training_data_current_batch_idx * training_elements_per_epoch : (training_data_current_batch_idx + 1) * training_elements_per_epoch]
            train_val_split = int(np.floor(len(data_this_epoch) * self.config.data.validation_split))
            self.train_indices = data_this_epoch[train_val_split:]
            self.valid_indices = data_this_epoch[:train_val_split]
            train_sampler = SubsetRandomSampler(self.train_indices)
            valid_sampler = SubsetRandomSampler(self.valid_indices)
            self.train_loader = DataLoader(self.train_dataset,
                                           batch_size=self.config.training.batch_size,
                                           sampler=train_sampler)
            self.valid_loader = DataLoader(self.train_dataset,
                                           batch_size=self.config.training.batch_size,
                                           sampler=valid_sampler)
        else:
            # Use separate validation dataset for validation
            # train data
            self.train_indices = self.overall_train_data_indices[training_data_current_batch_idx * training_elements_per_epoch : (training_data_current_batch_idx + 1) * training_elements_per_epoch]
            train_sampler = SubsetRandomSampler(self.train_indices)
            self.train_loader = DataLoader(self.train_dataset,
                                           batch_size=self.config.training.batch_size,
                                           sampler=train_sampler)
            # validation data is already subsampled via dataset constructor
            self.valid_indices = list(range(len(self.valid_dataset)))
            valid_sampler = SubsetRandomSampler(self.valid_indices)
            self.valid_loader = DataLoader(self.valid_dataset,
                                           batch_size=self.config.training.batch_size,
                                           sampler=valid_sampler)


    def _create_metric(self,
                       name: str,
                       available_metrics: Dict):
        if name not in list(available_metrics.keys()):
            raise ValueError(
                f"Unsupported metric: {name}. Must be one of "
                f"{list(available_metrics.keys())}"
            )
        if name == 'mse':
            return available_metrics[name]()
        if name == 'weighted_mse':
            return available_metrics[name]()
        if name == 'mpjl2':
            return available_metrics[name]()
        if name in ['mpjpe', 'mpvpe']:
            return available_metrics[name](
                self.target_body_type,
                self.config['general']['body_model_location'],
                self.config['general']['gender'],
                self.device
            )


    def _train_loop(self,
                    loss_fn,
                    metrics,
                    metric_weights = None,
                    tqdm_overwrite: TqdmUsage = None
                    ) -> Tuple[float, torch.Tensor]:
        if tqdm_overwrite is None:
            tqdm_overwrite = self.tqdm_usage
        if tqdm_overwrite != TqdmUsage.OFF:
            if tqdm_overwrite == TqdmUsage.SHELL:
                from tqdm import tqdm
            elif tqdm_overwrite == TqdmUsage.NOTEBOOK:
                from tqdm.notebook import tqdm
        size = len(self.train_loader.dataset)
        batch_size = self.train_loader.batch_size
        number_batches = size / batch_size
        total_loss = 0
        total_metrics = torch.tensor([0 for _ in metrics], dtype=torch.float32)
        for X, y in self.train_loader if tqdm_overwrite == TqdmUsage.OFF else tqdm(self.train_loader, desc="Training"):
            pred = self.model(X)
            try:
                loss = loss_fn(pred, y, metric_weights)
            except TypeError:
                loss = loss_fn(pred, y)
            total_loss += loss.item()

            # calculate metrics
            with torch.no_grad():
                for idx, metric in enumerate(metrics):
                    try:
                        total_metrics[idx] += metric(pred, y, metric_weights).item()
                    except TypeError:
                        total_metrics[idx] += metric(pred, y).item()
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        return total_loss / number_batches, total_metrics / number_batches


    def _valid_loop(self,
                    loss_fn,
                    metrics,
                    metric_weights = None,
                    tqdm_overwrite: TqdmUsage = None
                    ) -> Tuple[float, torch.Tensor]:
        if tqdm_overwrite is None:
            tqdm_overwrite = self.tqdm_usage
        if tqdm_overwrite != TqdmUsage.OFF:
            if tqdm_overwrite == TqdmUsage.SHELL:
                from tqdm import tqdm
            elif tqdm_overwrite == TqdmUsage.NOTEBOOK:
                from tqdm.notebook import tqdm
        num_batches = len(self.valid_loader)
        valid_loss = 0
        total_metrics = torch.tensor([0 for _ in metrics], dtype=torch.float32)

        with torch.no_grad():
            for X, y in self.train_loader if tqdm_overwrite == TqdmUsage.OFF else tqdm(self.valid_loader, desc="Validation"):
                pred = self.model(X)
                try:
                    valid_loss += loss_fn(pred, y, metric_weights).item()
                except TypeError:
                    valid_loss += loss_fn(pred, y).item()

                for idx, metric in enumerate(metrics):
                    try:
                        total_metrics[idx] += metric(pred, y, metric_weights).item()
                    except TypeError:
                        total_metrics[idx] += metric(pred, y).item()
        
        return valid_loss / num_batches, total_metrics / num_batches


    def _train_pass(self,
                    X: DotMap,
                    y: DotMap,
                    loss_fn: Callable,
                    metrics: List[str],
                    loss_fn_metrics: Callable,
                    fixed_input: DotMap,
                    fixed_target: DotMap,
                    n_training_samples: int,
                    normalizer = None
                    ) -> Tuple[torch.tensor, float, torch.tensor]:
        """
            Training for one batch, used by shape and pose trainers

            Params
            ------
                X (DotMap): Input parameters as dictionary with keys 'poses', 'trans' and 'betas'
                y (DotMap): Target parameters as dictionary with keys 'poses', 'trans' and 'betas'.
                loss_fn (Callable): Loss function
                metrics (List[str]): Metrics that should be evaluated
                loss_fn_metrics (Callable): Loss function with which the metrics should be calculated.
                fixed_input (DotMap): Input parameters that should be fixed for the whole batch. Keys 'poses', 'trans' and 'betas'
                fixed_target (DotMap): Target parameters that should be fixed for the whole batch. Keys 'poses', 'trans' and 'betas'
                n_training_samples (int): Number of training samples the complete training dataset consists of.
                normalizer:
                    Optional, normalization class that implements .transform() and .inverse_transform() interface.
                    Defaults to None.

            Returns
            -------
                torch.tensor: The loss that can be used to update network weights
                float: Loss value of this batch, already weighted by number of samples in the batch
                torch.tensor: The value for each metric, already weighted by number of samples in the batch
        """
        self.model.train()
        batch_loss = 0.0
        num_metrics = len(self.config.training.metrics) * len(self.config.training.metrics_reduction_mode)
        if self.config.training.calculate_metrics_for_ground_truth and not self.config.data.use_random_training_data:
            num_metrics *= 2
        batch_metrics = torch.zeros((num_metrics,), dtype=torch.float32)
        bs = len(X[self.config.conversion.parameter])
        batch_weight = bs / n_training_samples
        input_params = DotMap({
            'poses': X.poses if fixed_input.poses is None else fixed_input.poses.repeat(bs, 1),
            'trans': X.trans if fixed_input.trans is None else fixed_input.trans.repeat(bs, 1),
            'betas': X.betas[..., :self.config.data.n_shape_components] if fixed_input.betas is None else fixed_input.betas.repeat(bs, 1)
        }, _dynamic=False)
        network_input = X[self.config.conversion.parameter]
        if self.config.conversion.parameter == 'betas':
            network_input = network_input[..., :self.config.data.n_shape_components]
        pred = self.model(network_input)
        if normalizer is not None:
            pred = normalizer.inverse_transform(pred)

        predicted_params = DotMap({
            'poses': y.poses if fixed_target.poses is None else fixed_target.poses.repeat(bs, 1),
            'trans': y.trans if fixed_target.trans is None else fixed_target.trans.repeat(bs, 1),
            'betas': y.betas if fixed_target.betas is None else fixed_target.betas.repeat(bs, 1)
        }, _dynamic=False)
        predicted_params[self.config.conversion.parameter] = pred

        if self.config.training.calculate_metrics_for_ground_truth and not self.config.data.use_random_training_data:
            gt_params = DotMap({
                'poses': y.poses if fixed_target.poses is None else fixed_target.poses.repeat(bs, 1),
                'trans': y.trans if fixed_target.trans is None else fixed_target.trans.repeat(bs, 1),
                'betas': y.betas[..., :self.config.data.n_shape_components] if fixed_target.betas is None else fixed_target.betas.repeat(bs, 1)
            }, _dynamic=False)
        loss = loss_fn(input_params,
                       predicted_params,
                       loss_type=self.config.training.loss_fn,
                       reduction_mode=self.config.training.loss_reduction_mode,
                       param_names=self.config.conversion.parameter)
        batch_loss = loss.item() * batch_weight

        # calculate metrics
        i = 0
        with torch.no_grad():
            for metric in metrics:
                for red_mode in self.config.training.metrics_reduction_mode:
                    batch_metrics[i] += loss_fn_metrics(input_params,
                                                        predicted_params,
                                                        loss_type=metric,
                                                        reduction_mode=red_mode,
                                                        param_names=self.config.conversion.parameter
                                                        ).item() * batch_weight
                    if self.config.training.calculate_metrics_for_ground_truth and not self.config.data.use_random_training_data:
                        batch_metrics[i+(num_metrics//2)] += loss_fn_metrics(input_params,
                                                                             gt_params,
                                                                             loss_type=metric,
                                                                             reduction_mode=red_mode,
                                                                             param_names=self.config.conversion.parameter
                                                                             ).item() * batch_weight
                    i += 1
        return loss, batch_loss, batch_metrics


    def _valid_pass(self,
                    X: DotMap,
                    y: DotMap,
                    loss_fn: Callable,
                    metrics,
                    loss_fn_metrics: Callable,
                    fixed_input: DotMap,
                    fixed_target: DotMap,
                    n_validation_samples: int,
                    normalizer = None
                    ) -> Tuple[float, torch.tensor]:
        self.model.eval()
        batch_loss = 0.0
        num_metrics = len(self.config.training.metrics) * len(self.config.training.metrics_reduction_mode)
        if self.config.training.calculate_metrics_for_ground_truth:
            num_metrics *= 2
        batch_metrics = torch.zeros(num_metrics, dtype=torch.float32)
        bs = len(X[self.config.conversion.parameter])
        batch_weight = bs / n_validation_samples

        input_params = DotMap({
            'poses': X.poses if fixed_input.poses is None else fixed_input.poses.repeat(bs, 1),
            'trans': X.trans if fixed_input.trans is None else fixed_input.trans.repeat(bs, 1),
            'betas': X.betas[..., :self.config.data.n_shape_components] if fixed_input.betas is None else fixed_input.betas.repeat(bs, 1)
        }, _dynamic=False)
        network_input = X[self.config.conversion.parameter]
        if self.config.conversion.parameter == 'betas':
            network_input = network_input[..., :self.config.data.n_shape_components]
        pred = self.model(network_input)
        if normalizer is not None:
            pred = normalizer.inverse_transform(pred)

        predicted_params = DotMap({
            'poses': y.poses if fixed_target.poses is None else fixed_target.poses.repeat(bs, 1),
            'trans': y.trans if fixed_target.trans is None else fixed_target.trans.repeat(bs, 1),
            'betas': y.betas if fixed_target.betas is None else fixed_target.betas.repeat(bs, 1)
        }, _dynamic=False)
        predicted_params[self.config.conversion.parameter] = pred

        if self.config.training.calculate_metrics_for_ground_truth:
            gt_params = DotMap({
                'poses': y.poses if fixed_target.poses is None else fixed_target.poses.repeat(bs, 1),
                'trans': y.trans if fixed_target.trans is None else fixed_target.trans.repeat(bs, 1),
                'betas': y.betas[..., :self.config.data.n_shape_components] if fixed_target.betas is None else fixed_target.betas.repeat(bs, 1)
            }, _dynamic=False)
        loss = loss_fn(input_params,
                       predicted_params,
                       loss_type=self.config.training.loss_fn,
                       reduction_mode=self.config.training.loss_reduction_mode,
                       param_names=self.config.conversion.parameter)
        batch_loss = loss.item() * batch_weight

        # calculate metrics
        i = 0
        with torch.no_grad():
            for metric in metrics:
                for red_mode in self.config.training.metrics_reduction_mode:
                    batch_metrics[i] += loss_fn_metrics(input_params,
                                                        predicted_params,
                                                        loss_type=metric,
                                                        reduction_mode=red_mode,
                                                        param_names=self.config.conversion.parameter
                                                        ).item() * batch_weight
                    if self.config.training.calculate_metrics_for_ground_truth:
                        batch_metrics[i+(num_metrics//2)] += loss_fn_metrics(input_params,
                                                                             gt_params,
                                                                             loss_type=metric,
                                                                             reduction_mode=red_mode,
                                                                             param_names=self.config.conversion.parameter
                                                                             ).item() * batch_weight
                    i += 1
        return batch_loss, batch_metrics

    def _write_log(self, path, train_progress):
        with open(path, "wb") as file:
            pickle.dump(train_progress, file)
