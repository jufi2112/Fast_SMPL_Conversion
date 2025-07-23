# Copyright (c) 2025 Julien Fischer, Chair of Computer Graphics and Visualization, TUD Dresden University of Technology.
# All rights reserved.
# Licensed under the MIT License. See LICENSE in the project root for license information.

import os
import math
import torch
import argparse
import numpy as np

from tqdm import tqdm
from dotmap import DotMap
from os import path as osp
from time import perf_counter
from datetime import datetime
from tqdm.notebook import tqdm as tqdm_notebook
from smpl_conversion.factories import ConversionModelFactory
from typing import Union, Dict, Callable, List, Tuple, Optional
from smpl_conversion.training.base_trainers import AbstractNNTrainer
from smpl_conversion.losses import CorrespondenceLoss, DirectParameterLoss
from smpl_conversion.utils.experiment_directory_manager import ExperimentDirectoryManager
from smpl_conversion.models.combined import assemble_combined_network_input, split_network_output
from smpl_conversion.utils.enum_configurations import (
    TqdmUsage,
    ExperimentType,
    BodyModelType,
)
from smpl_conversion.utils.random_parameter_sampling import (
    sample_random_translation,
    sample_random_shape,
    sample_random_pose
)


class CombinedTrainer(AbstractNNTrainer):
    def __init__(self,
                 config: Optional[Union[Dict, str]] = None,
                 checkpoint: Optional[Union[Dict, str]] = None,
                 finetune: Optional[bool] = False
                 ):
        """
            Class to train a combined (i.e. multiple parameters jointly) conversion network for the parameter conversion from one
            parameterized human body model (SMPL / SMPL+H / SMPL-X / SUPR) to another.

            Params
            ------
                config (dict or str):
                    Optional, training configuration. Can either be an already loaded dict or a path to a .yaml file.
                    Defaults to None (will then use the provided checkpoint_file). Specifying both or none of the parameters is an error.
                checkpoint (dict or str):
                    Optional, a checkpoint (or path to it) from which the training should be continued. Defaults to None
                    (will then use the provided config and start training from scratch). Specifying both or none of the parameters is an error.
                finetune (bool):
                    Whether finetuning on an existing dataset should be performed. Optional, defaults to False
        """
        if checkpoint is None and config is None:
            raise ValueError("Error: Either config or checkpoint has to be specified!")
        self.finetune = finetune
        if self.finetune:
            if config is None or checkpoint is None:
                raise ValueError(f"For finetuning, both a config and checkpoint must be provided")
            if isinstance(checkpoint, dict):
                self.ckpt = checkpoint
            elif isinstance(checkpoint, str):
                self.ckpt = torch.load(checkpoint)
            else:
                raise TypeError(f"Got unexpected type for argument checkpoint: {type(checkpoint)}")
        else:
            if checkpoint is not None:
                if config is not None:
                    raise ValueError("Specified both config and checkpoint_file!")
                print("Creating CombinedTrainer from existing checkpoint...")
                if isinstance(checkpoint, dict):
                    self.ckpt = checkpoint
                elif isinstance(checkpoint, str):
                    self.ckpt = torch.load(checkpoint)
                else:
                    raise TypeError(f"Expected argument 'checkpoint' to be of type 'dict' or 'str' but got {type(checkpoint)}")
                if 'is_sanitized' in self.ckpt.keys() and self.ckpt['is_sanitized'] == True:
                    raise ValueError("Error: The provided checkpoint has been sanitized, unable to resume training from it.")
                config = self.ckpt['train_progress']['config']
            else:
                self.ckpt = None
        super().__init__(config)
        if "metrics_reduction_mode" in self.config.training.keys():
            if not isinstance(self.config.training.metrics_reduction_mode, list):
                self.config.training.metrics_reduction_mode = [self.config.training.metrics_reduction_mode]
        else:
            raise ValueError("Please provide reduction mode for metrics: metrics_reduction_mode")
        self._log_text(f"Network input:  {str(self.input_rotation_representation)}")
        self._log_text(f"Network output: {str(self.output_rotation_representation)}")


    def _create_model(self):
        """
            Creates a new conversion model instance.
        """
        self.model = ConversionModelFactory.create_conversion_model(self.config,
                                                                    self.device)
        print(self.model)


    def fit(self,
            clear_old_model: bool = True
            ) -> Dict:
        """
            Starts the training process and fits the model to the data. The training is carried out according to the
            configuration provided to the constructor (including options for checkpointing). If a checkpoint was
            given to the constructor, this continues the training of this checkpoint (clear_old_model will not remove
            checkpoint progress).

            Params
            ------
                clear_old_model (bool):
                    Whether any existing previous model should be overwritten.
                    Defaults to True.

            Returns
            -------
                Dict:
                    A dictionary containing the training progress.
        """
        train_start_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        train_progress = None
        lowest_valid_loss = math.inf
        previous_training_time = 0
        if self.model is None or clear_old_model:
            self._create_model()
        self._create_optimizer_and_lr_scheduler()

        # When finetuning, only load model state dict but nothing else
        if self.finetune:
            self.model.load_state_dict(self.ckpt['model_state_dict'])
            self._log_text("Loaded model weights from checkpoint")
        elif hasattr(self, 'ckpt') and self.ckpt is not None:
            (
                lowest_valid_loss,
                previous_training_time,
                train_start_time,
                train_progress
            ) = self._load_checkpoint_states()
            self._log_text(f"Sucessfully loaded states and statistics of checkpoint, who's training started at {train_start_time} and which was already trained for {previous_training_time} seconds.")

        if self.input_body_type == BodyModelType.SUPR or self.target_body_type == BodyModelType.SUPR:
            if self.config.conversion.supr_is_constrained:
                self._log_text("Using constrained SUPR body model")
            else:
                self._log_text("Using unconstrained SUPR body model")
        if self.config.data.train_on_random_shape_data:
            self._log_text(
                "Will use randomly sampled shape parameters during training.\n"
                "Validation is done on AMASS shape parameters"
            )
        if self.config.data.train_on_random_pose_data:
            self._log_text(
                "Will use randomly sampled pose parameters during training.\n"
                "Validation is done on AMASS pose parameters"
            )
        elif self.config.data.train_on_random_global_orient_data:
            self._log_text(
                "Will use randomly sampled global orientation parameters during training.\n"
                "Validation is done on AMASS global orientation parameters"
            )
        if self.config.data.train_on_random_trans_data:
            self._log_text(
                "Will use randomly sampled translation parameters during training.\n"
                "Validation is done on AMASS translation parameters"
            )
        if self.config.training.calculate_metrics_for_ground_truth and self.train_dataset.is_unsupervised:
            self._log_text("Will not be performing evaluation on ground truth data, as the loaded dataset does not contain entries for the target model type.")
        if self.config.training.loss_fn == 'vertex_edge':
            self._log_text(f"Will use combined vertex and edge loss weighted by:\n   Vertex: {self.config.training.loss_fn_weights.get('vertex', 1)}\n   Edge: {self.config.training.loss_fn_weights.get('edge', 1)}")
        if self.finetune:
            fname = 'finetuned_' + train_start_time
        else:
            fname = 'combined_' + train_start_time
        if self.config.training.loss_fn != 'direct':
            loss_fn_name = f"{self.config.training.loss_fn}{'' if self.config.training.loss_fn == 'mse' else '_'+self.config.training.loss_reduction_mode}"
            loss_fn = CorrespondenceLoss(self.input_body_type,
                                         self.target_body_type,
                                         self.config.general.gender,
                                         self.config.general.body_model_location,
                                         self.config.general.transfer_file_location,
                                         self.config.training.batch_size,
                                         self.config.data.n_shape_components,
                                         self.device,
                                         self.config.conversion.supr_is_constrained,
                                         self.input_rotation_representation,
                                         self.output_rotation_representation
                                         )
            loss_fn_metrics = loss_fn
        else:
            loss_fn_name = "direct"
            loss_fn = DirectParameterLoss()
            loss_fn_metrics = CorrespondenceLoss(self.input_body_type,
                                                 self.target_body_type,
                                                 self.config.general.gender,
                                                 self.config.general.body_model_location,
                                                 self.config.general.transfer_file_location,
                                                 self.config.training.batch_size,
                                                 self.config.data.n_shape_components,
                                                 self.device,
                                                 self.config.conversion.supr_is_constrained,
                                                 self.input_rotation_representation,
                                                 self.output_rotation_representation
                                                 )
        metrics = self.config.training.metrics

        if not osp.isdir(self.config.training.log_dir):
            os.makedirs(self.config.training.log_dir)
        if not osp.isdir(self.config.training.checkpoint_dir):
            os.makedirs(self.config.training.checkpoint_dir)

        if train_progress is None:
            train_progress = {
                'epochs': {},
                'loss_fn': loss_fn_name,
                'optimizer': type(self.optimizer).__name__,
                'start': train_start_time,
                'config': self.config,
                'model_architecture': str(self.model),
                'train_indices': None,
                'valid_indices': None
            }

        if self.tqdm_usage == TqdmUsage.NOTEBOOK:
            loop = tqdm_notebook
        elif self.tqdm_usage == TqdmUsage.SHELL:
            loop = tqdm

        start_epoch = len(list(train_progress['epochs'].keys()))
        training_time_start = perf_counter()
        for epoch in range(start_epoch, self.config.training.epochs) if self.tqdm_usage == TqdmUsage.OFF else loop(range(start_epoch, self.config.training.epochs), desc="Finetuning" if self.finetune else "Training"):
            self._recreate_random_data_loaders(self.config.data.subsampling_factor_train if "subsampling_factor_train" in self.config.data.keys() else self.config.data.subsampling_factor,
                                               epoch)
            print_current_epoch = False
            if epoch % self.config.training.epoch_report_interval == 0 or (epoch+1) == self.config.training.epochs or epoch == start_epoch:
                print_current_epoch = True
                self._log_text(f"Epoch {epoch+1:03} / {self.config.training.epochs:03}")

            train_loss, train_metrics = self._train_loop(
                loss_fn,
                self.config.training.loss_fn_weights,
                metrics,
                loss_fn_metrics,
                tqdm_overwrite=TqdmUsage.OFF if not print_current_epoch else None,
                normalizer=None
            )

            valid_loss, valid_metrics = self._valid_loop(
                loss_fn,
                self.config.training.loss_fn_weights,
                metrics,
                loss_fn_metrics,
                tqdm_overwrite=TqdmUsage.OFF if not print_current_epoch else None,
                normalizer=None
            )

            if isinstance(self.optimizer, torch.optim.Adam) and not self.lr_scheduler:
                lr = self.optimizer.param_groups[0]['lr']
            else:
                lr = self.lr_scheduler._last_lr[0] if epoch > 0 else self.config.training.lr

            if print_current_epoch:
                text = f"Train loss: {train_loss:>7f} | Valid loss: " \
                    f"{valid_loss:>7f} | LR {lr:>7f}"
                self._log_text(text)

            # Update training log
            train_mets = [f"{met}_{red}" for met in metrics for red in self.config.training.metrics_reduction_mode]
            valid_mets = [f"{met}_{red}" for met in metrics for red in self.config.training.metrics_reduction_mode]
            if self.config.training.calculate_metrics_for_ground_truth and not self.train_dataset.is_unsupervised:
                valid_mets.extend([f"gt_{met}" for met in valid_mets])
                if not self.config.data.train_on_random_shape_data:
                    train_mets.extend([f"gt_{met}" for met in train_mets])
            train_progress['epochs'][epoch+1] = {
                'train': {
                    'loss': train_loss,
                    'metrics': {
                        met: train_metrics[idx].item()
                        for idx, met in enumerate(train_mets)
                    }
                },
                'validation': {
                    'loss': valid_loss,
                    'metrics': {
                        met: valid_metrics[idx].item()
                        for idx, met in enumerate(valid_mets)
                    }
                },
                'lr': lr,
                'training_time_s': perf_counter() - training_time_start + previous_training_time
            }

            if self.lr_scheduler:
                if isinstance(self.lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.lr_scheduler.step(valid_loss)
                else:
                    self.lr_scheduler.step()

            # Dump log
            self._write_log(
                osp.join(self.config.training.log_dir, f"{fname}.pkl"),
                train_progress
            )

            self._save_model_checkpoint(valid_loss, lowest_valid_loss,
                                        epoch+1, train_progress, fname)
            if valid_loss < lowest_valid_loss:
                lowest_valid_loss = valid_loss

            # Early stopping:
            if self.config.training.enable_early_stopping:
                if lr < self.config.training.lr_early_stopping_threshold:
                    if not print_current_epoch:
                        text = f"Epoch {epoch+1:03} / " \
                            f"{self.config.training.epochs} Train loss: " \
                            f"{train_loss:>7f} | Valid loss: " \
                            f"{valid_loss:>7f} | LR: {lr:>7f}\n"
                        self._log_text(text)
                    self._log_text(f"Early stopped training after {epoch+1} epochs.")
        last_epoch = sorted(train_progress['epochs'].keys())[-1]
        self._log_text(f"{'Finetuning' if self.finetune else 'Training'} finished after {train_progress['epochs'][last_epoch]['training_time_s']:.2f} seconds")
        return train_progress


    def predict(self,
                X: DotMap,
                checkpoint: str = None
                ) -> DotMap:
        """
            Converts the given parameters. Currently not implemented, please use the functionality in
            inference/combined_prediction.py

            Params
            ------
                X (DotMap):
                    Dict of original parameters with keys 'trans', 'betas',
                    and 'poses'
                checkpoint (str):
                    Optional, checkpoint location whose weights should
                    be used for prediction. Defaults to None
            Returns:
                DotMap:
                    Converted parameters. Keys are 'trans', 'betas', 'poses'
        """
        raise NotImplementedError(
            "Prediction not supported in CombinedTraining class, use "
            "CombinedPredictor class from inference/combined_prediction.py instead!"
        )


    def _train_loop(self,
                    loss_fn: Callable,
                    loss_fn_weights: Dict,
                    metrics: List[str],
                    loss_fn_metrics: Callable,
                    tqdm_overwrite: TqdmUsage = None,
                    normalizer: Callable = None
                    ) -> Tuple[float, torch.Tensor]:
        """
            Custom training loop called from inside the fit() method.

            Params
            ------
                loss_fn (Callable):
                    Loss function instance to use.
                metrics (list of str):
                    Metrics that should be used for evaluation.
                loss_fn_metrics (Callable):
                    Loss function that should be used for the above metrics.
                tqdm_overwrite (TqdmUsage):
                    A value with which to overwrite the default settings of how to use tqdm.
                    Defaults to None.
                normalizer (Callable):
                    Normalization class instance that implements .transform() and .inverse_transform() interface.
                    Defaults to None

            Returns
            -------
                float:
                    Train loss over the whole epoch. As earlier batches are usually evaluated on 'worse' weights than later batches,
                    train loss might be higher than validation loss (which is evaluated on the weights of the last batch).
                torch.Tensor:
                    Evaluation metrics over the whole epoch. Same implications as for the loss value.
        """
        if tqdm_overwrite is None:
            tqdm_overwrite = self.tqdm_usage
        if tqdm_overwrite != TqdmUsage.OFF:
            if tqdm_overwrite == TqdmUsage.SHELL:
                loop = tqdm
            elif tqdm_overwrite == TqdmUsage.NOTEBOOK:
                loop = tqdm_notebook
        num_metrics = len(self.config.training.metrics) * len(self.config.training.metrics_reduction_mode)
        train_on_random_data = self.config.data.train_on_random_shape_data or self.config.data.train_on_random_trans_data or self.config.data.train_on_random_pose_data or self.config.data.train_on_random_global_orient_data
        if self.config.training.calculate_metrics_for_ground_truth and not train_on_random_data and not self.train_dataset.is_unsupervised:
            num_metrics *= 2
        total_loss = 0.0
        total_metrics = torch.zeros((num_metrics,), dtype=torch.float32).requires_grad_(False)

        if self.config.data.train_on_random_shape_data:
            uniform_dist = DotMap(self.config.data.random_shape_data.uniform,
                                  _dynamic=False)
            normal_dist = DotMap(self.config.data.random_shape_data.normal,
                                 _dynamic=False)
            random_shape = sample_random_shape(len(self.train_indices),
                                               self.config.data.n_shape_components,
                                               uniform_dist if self.config.data.random_shape_data.random_mode == 'uniform' else None,
                                               normal_dist if self.config.data.random_shape_data.random_mode == 'normal' else None)
            random_shape = random_shape.to(self.config.general.device_dataset)
        else:
            random_shape = None
        if self.config.data.train_on_random_pose_data:
            random_poses = sample_random_pose(len(self.train_indices), self.input_body_type, self.np_rng)
            random_poses = random_poses.to(self.config.general.device_dataset)
        elif self.config.data.train_on_random_global_orient_data:
            random_poses = sample_random_pose(len(self.train_indices), self.input_body_type, self.np_rng, 1)
            random_poses = random_poses.to(self.config.general.device_dataset)
        else:
            random_poses = None
        if self.config.data.train_on_random_trans_data:
            uniform_dist = DotMap(self.config.data.random_trans_data.uniform, _dynamic=False)
            normal_dist = DotMap(self.config.data.random_trans_data.normal, _dynamic=False)
            random_trans = sample_random_translation(len(self.train_indices),
                                                     uniform_dist if self.config.data.random_trans_data.random_mode == "uniform" else None,
                                                     normal_dist if self.config.data.random_trans_data.random_mode == "normal" else None
                                                     )
            random_trans = random_trans.to(self.config.general.device_dataset)
        else:
            random_trans = None
        for batch_idx, (X, y) in enumerate(self.train_loader) if tqdm_overwrite == TqdmUsage.OFF else enumerate(loop(self.train_loader, desc="Training", position=1, leave=False)):
            if self.config.data.train_on_random_shape_data:
                X.betas = random_shape[batch_idx * self.config.training.batch_size : (batch_idx+1) * self.config.training.batch_size, ...].to(self.device)
            if self.config.data.train_on_random_pose_data:
                X.poses = random_poses[batch_idx * self.config.training.batch_size : (batch_idx+1) * self.config.training.batch_size, ...].to(self.device)
            elif self.config.data.train_on_random_global_orient_data:
                X.poses[:, :3] = random_poses[batch_idx * self.config.training.batch_size : (batch_idx+1) * self.config.training.batch_size, :3].to(self.device)
            if self.config.data.train_on_random_trans_data:
                X.trans = random_trans[batch_idx * self.config.training.batch_size : (batch_idx+1) * self.config.training.batch_size, ...].to(self.device)
            loss, loss_nbr, metrics_nbr = self._train_pass(X, y, loss_fn, loss_fn_weights, metrics,
                                                           loss_fn_metrics, len(self.train_indices),
                                                           not self.evaluate_train_data_after_last_batch,
                                                           normalizer
                                                           )
            if self.check_for_nan and torch.isnan(loss):
                raise RuntimeError("Got a loss tensor of nan!")
            total_loss += loss_nbr
            # if self.evaluate_train_data_after_last_batch is True, metrics_nbr will be zero tensor
            total_metrics += metrics_nbr

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        # Check whether we want to calculate training metrics after last batch of training
        # Evaluates the model on the same randomly sampled parameters as it was trained on in this epoch
        if self.evaluate_train_data_after_last_batch:
            total_loss = 0
            total_metrics.zero_()
            for batch_idx, (X, y) in enumerate(self.train_loader) if tqdm_overwrite == TqdmUsage.OFF else enumerate(loop(self.train_loader, desc="Training Evaluation", position=1, leave=False)):
                if self.config.data.train_on_random_shape_data:
                    X.betas = random_shape[batch_idx * self.config.training.batch_size : (batch_idx+1) * self.config.training.batch_size, ...].to(self.device)
                    X.betas.requires_grad_(False)
                if self.config.data.train_on_random_pose_data:
                    X.poses = random_poses[batch_idx * self.config.training.batch_size : (batch_idx+1) * self.config.training.batch_size, ...].to(self.device)
                    X.poses.requires_grad_(False)
                if self.config.data.train_on_random_trans_data:
                    X.trans = random_trans[batch_idx * self.config.training.batch_size : (batch_idx+1) * self.config.training.batch_size, ...].to(self.device)
                    X.trans.requires_grad_(False)
                loss, metrics_nbr = self._valid_pass(X, y, loss_fn, loss_fn_weights, metrics,
                                                     loss_fn_metrics, len(self.train_indices),
                                                     normalizer, train_on_random_data)
                total_loss += loss
                total_metrics += metrics_nbr
        return total_loss, total_metrics


    def _valid_loop(self,
                    loss_fn: Callable,
                    loss_fn_weights: Dict,
                    metrics: List[str],
                    loss_fn_metrics: Callable,
                    tqdm_overwrite: TqdmUsage = None,
                    normalizer: Callable = None
                    ) -> Tuple[float, torch.Tensor]:
        """
            Custom validation loop called from inside the fit() method.

            Params
            ------
                loss_fn (Callable):
                    Loss function instance to use.
                loss_fn_weights (Dict):
                    If loss_fn is 'vertex_edge', this dict should contain the weights with which to
                    weight the respective losses.
                metrics (list of str):
                    Metrics that should be used for evaluation.
                loss_fn_metrics (Callable):
                    Loss function that should be used for the above metrics.
                tqdm_overwrite (TqdmUsage):
                    A value with which to overwrite the default settings of how to use tqdm.
                    Defaults to None.
                normalizer (Callable):
                    Normalization class instance that implements .transform() and .inverse_transform() interface.
                    Defaults to None

            Returns
            -------
                float:
                    Validation loss. As this is evaluated only on the weights of the last training batch,
                    validation loss might be lower than training loss (which is evaluated on the weights of all batches).
                torch.Tensor:
                    Evaluation metrics. Same implications as for the loss value.
        """
        if tqdm_overwrite is None:
            tqdm_overwrite = self.tqdm_usage
        if tqdm_overwrite != TqdmUsage.OFF:
            if tqdm_overwrite == TqdmUsage.SHELL:
                loop = tqdm
            elif tqdm_overwrite == TqdmUsage.NOTEBOOK:
                loop = tqdm_notebook
        num_metrics = len(metrics) * len(self.config.training.metrics_reduction_mode)
        if self.config.training.calculate_metrics_for_ground_truth and not self.train_dataset.is_unsupervised:
            num_metrics *= 2
        valid_loss = 0.0
        total_metrics = torch.zeros((num_metrics,), dtype=torch.float32).requires_grad_(False)

        for X, y in self.valid_loader if tqdm_overwrite == TqdmUsage.OFF else loop(self.valid_loader, desc="Validation", position=1, leave=False):
            loss, metrics_nbr = self._valid_pass(X, y, loss_fn, loss_fn_weights, metrics,
                                                 loss_fn_metrics, len(self.valid_indices),
                                                 normalizer, False)
            valid_loss += loss
            total_metrics += metrics_nbr
        return valid_loss, total_metrics


    def _train_pass(self,
                    X: DotMap,
                    y: DotMap,
                    loss_fn: Callable,
                    loss_fn_weights: Dict,
                    metrics,
                    loss_fn_metrics: Callable,
                    n_training_samples: int,
                    calculate_metrics: bool,
                    normalizer: Callable = None,
                    ) -> Tuple[torch.Tensor, float, torch.Tensor]:
        """
            Training pass for one batch of data.

            Params
            ------
                X (DotMap):
                    Input parameters as dictionary with keys 'trans', 'betas', and 'poses'
                y (DotMap):
                    Target parameters as dictionary with keys 'trans', 'betas', and 'poses'
                loss_fn (Callable):
                    Loss function
                loss_fn_weights (Dict[str, float]):
                    If loss is 'vertex_edge', a dictionary that contains the weights for vertex
                    and edge loss.
                metrics (List[str]):
                    Metrics that should be evaluated
                loss_fn_metrics (Callable):
                    Loss function with which the metrics should be calculated
                n_training_samples (int):
                    Total amount of training samples that will be used in this epoch. Used for averaging
                    if the batch size changes over the course of the epoch (e.g. in the last batch).
                calculate_metrics (bool):
                    Whether metrics should be calculated.
                normalizer (Callable):
                    Normalization class instance that implements .transform() and .inverse_transform() interface.
                    Defaults to None.

            Returns
            -------
                torch.Tensor:
                    The loss that can be used to update network weights
                float:
                    Loss value of this batch, already weighted by the number of training samples in the batch
                torch.Tensor:
                    The value for each metric, already weighted by the number of training samples in the batch
        """
        self.model.train()
        train_on_random_data = self.config.data.train_on_random_shape_data or self.config.data.train_on_random_trans_data or self.config.data.train_on_random_pose_data or self.config.data.train_on_random_global_orient_data
        evaluate_on_gt = self.config.training.calculate_metrics_for_ground_truth and not train_on_random_data and not self.train_dataset.is_unsupervised
        batch_loss = 0.0
        num_metrics = len(self.config.training.metrics) * len(self.config.training.metrics_reduction_mode)
        if evaluate_on_gt:
            num_metrics *= 2
        batch_metrics = torch.zeros((num_metrics,), dtype=torch.float32)
        bs = len(X.poses)
        batch_weight = bs / n_training_samples
        input_params = DotMap({
            'trans': X.trans,
            'betas': X.betas[..., :self.config.data.n_shape_components],
            'poses': X.poses
        }, _dynamic=False)
        network_input = assemble_combined_network_input(self.config.data.n_shape_components,
                                                        input_dict=input_params)
        pred = self.model(network_input)
        if normalizer is not None:
            pred = normalizer.inverse_transform(pred)

        pred_trans, pred_betas, pred_poses = split_network_output(pred,
                                                                  self.config.data.n_shape_components)
        predicted_params = DotMap({
            'trans': pred_trans,
            'betas': pred_betas,
            'poses': pred_poses
        }, _dynamic=False)

        if evaluate_on_gt:
            # Ground truth data provided by SMPL_Conversion_Dataset are already
            # converted to output_rotation_representation and can be directly
            # provided to loss function (loss function converts it as required
            # to either rotation vectors or rotation matrices)
            gt_params = DotMap({
                'trans': y.trans,
                'betas': y.betas[..., :self.config.data.n_shape_components],
                'poses': y.poses
            }, _dynamic=False)
        if self.config.training.loss_fn == 'vertex_edge':
            vertex_loss = loss_fn(input_params,
                                  predicted_params,
                                  loss_type="vertex",
                                  reduction_mode="mean",
                                  param_names=self.config.conversion.parameter)
            edge_loss = loss_fn(input_params,
                                predicted_params,
                                loss_type="edge",
                                reduction_mode="mean",
                                param_names=self.config.conversion.parameter)
            vertex_weight = loss_fn_weights.get('vertex', 1)
            edge_weight = loss_fn_weights.get('edge', 1)
            loss = vertex_weight * vertex_loss + edge_weight * edge_loss
        else:
            loss = loss_fn(input_params,
                           predicted_params,
                           loss_type=self.config.training.loss_fn,
                           reduction_mode=self.config.training.loss_reduction_mode,
                           param_names=self.config.conversion.parameter)
        batch_loss = loss.item() * batch_weight

        # calculate metrics
        if calculate_metrics:
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
                        if evaluate_on_gt:
                            batch_metrics[i+(num_metrics//2)] += loss_fn_metrics(input_params,
                                                                                 gt_params,
                                                                                 loss_type=metric,
                                                                                 reduction_mode=red_mode,
                                                                                 param_names=self.config.conversion.parameter,
                                                                                 ).item() * batch_weight
                        i += 1
        return loss, batch_loss, batch_metrics


    def _valid_pass(self,
                    X: DotMap,
                    y: DotMap,
                    loss_fn: Callable,
                    loss_fn_weights: Dict,
                    metrics: List[str],
                    loss_fn_metrics: Callable,
                    n_validation_samples: int,
                    normalizer = None,
                    data_is_random: bool = False
                    ) -> Tuple[float, torch.Tensor]:
        """
            Validation pass for one batch of data. See _train_pass for parameters

            Params
            ------
                X (DotMap):
                    Input parameters as dictionary with keys 'trans', 'betas', and 'poses'
                y (DotMap):
                    Target parameters as dictionary with keys 'trans', 'betas', and 'poses'
                loss_fn (Callable):
                    Loss function
                loss_fn_weights (Dict):
                    If loss is 'vertex_edge', this dict should contain the weights with which to
                    weight the corresponding losses.
                metrics (List[str]):
                    Metrics that should be evaluated
                loss_fn_metrics (Callable):
                    Loss function with which the metrics should be calculated
                n_validation_samples (int):
                    Total amount of validation samples that will be used in this epoch.
                    Used for averaging if the batch size changes over the course of the epoch (e.g. in the last batch).
                normalizer (Callable):
                    Normalization class instance that implements .transform() and .inverse_transform()
                    interface. Defaults to None.
                data_is_random (bool):
                    Whether the provided data is random, such that no ground truth evaluation can be confirmed

            Returns
            -------
                float:
                    Loss value of this batch, already weighted by the number of validation samples in the batch
                torch.Tensor:
                    The value for each metric, already weighted by the number of validation samples in the batch
        """
        self.model.eval()
        with torch.no_grad():
            evaluate_on_gt = self.config.training.calculate_metrics_for_ground_truth and not self.train_dataset.is_unsupervised and not data_is_random
            batch_loss = 0.0
            num_metrics = len(metrics) * len(self.config.training.metrics_reduction_mode)
            if evaluate_on_gt:
                num_metrics *= 2
            batch_metrics = torch.zeros(num_metrics, dtype=torch.float32)
            bs = len(X.poses)
            batch_weight = bs / n_validation_samples

            input_params = DotMap({
                'trans': X.trans,
                'betas': X.betas[..., :self.config.data.n_shape_components],
                'poses': X.poses
            }, _dynamic=False)
            network_input = assemble_combined_network_input(self.config.data.n_shape_components,
                                                            input_dict=input_params)
            pred = self.model(network_input)
            if normalizer is not None:
                pred = normalizer.inverse_transform(pred)

            pred_trans, pred_betas, pred_poses = split_network_output(pred,
                                                                      self.config.data.n_shape_components)
            predicted_params = DotMap({
                'trans': pred_trans,
                'betas': pred_betas,
                'poses': pred_poses
            }, _dynamic=False)

            if evaluate_on_gt:
                gt_params = DotMap({
                    'trans': y.trans,
                    'betas': y.betas[..., :self.config.data.n_shape_components],
                    'poses': y.poses
                }, _dynamic=False)

            if self.config.training.loss_fn == 'vertex_edge':
                vertex_loss = loss_fn(input_params,
                                      predicted_params,
                                      loss_type="vertex",
                                      reduction_mode="mean",
                                      param_names=self.config.conversion.parameter)
                edge_loss = loss_fn(input_params,
                                    predicted_params,
                                    loss_type="edge",
                                    reduction_mode="mean",
                                    param_names=self.config.conversion.parameter)
                vertex_weight = loss_fn_weights.get('vertex', 1)
                edge_weight = loss_fn_weights.get('edge', 1)
                loss = vertex_weight * vertex_loss + edge_weight * edge_loss
            else:
                loss = loss_fn(input_params,
                               predicted_params,
                               loss_type=self.config.training.loss_fn,
                               reduction_mode=self.config.training.loss_reduction_mode,
                               param_names=self.config.conversion.parameter)
            batch_loss = loss.item() * batch_weight

            # calculate metrics
            i = 0
            for metric in metrics:
                for red_mode in self.config.training.metrics_reduction_mode:
                    batch_metrics[i] += loss_fn_metrics(input_params,
                                                        predicted_params,
                                                        loss_type=metric,
                                                        reduction_mode=red_mode,
                                                        param_names=self.config.conversion.parameter
                                                        ).item() * batch_weight
                    if evaluate_on_gt:
                        batch_metrics[i+(num_metrics//2)] += loss_fn_metrics(input_params,
                                                                             gt_params,
                                                                             loss_type=metric,
                                                                             reduction_mode=red_mode,
                                                                             param_names=self.config.conversion.parameter
                                                                             ).item() * batch_weight
                    i += 1
        return batch_loss, batch_metrics


    def _load_checkpoint_states(self) -> Tuple[float, float, float, Dict]:
        """
            Loads a checkpoint that has previously been provided to the constructor.
            This includes the random states of torch and numpy, as well as
            the states of model, optimizer, and, if applicable, learning rate scheduling.

            Returns
            -------
                float:
                    lowest validation loss
                float:
                    total training time (s)
                float:
                    training start time (timestamp)
                Dict:
                    training progress
        """
        if not hasattr(self, 'ckpt') or self.ckpt is None:
            raise ValueError("Trying to load a checkpoint even though no checkpoint file has been provided during initialization")

        if 'is_sanitized' in self.ckpt.keys():
            print("The provided checkpoint has been sanitized and can only be used for predictions, not for training.")
            is_sanitized = True
        else:
            is_sanitized = False
        if 'torch_random_state' in self.ckpt.keys():
            torch.set_rng_state(self.ckpt['torch_random_state'])
        else:
            print("Could not find torch random state in checkpoint.")
        if 'numpy_random_state' in self.ckpt.keys():
            np.random.set_state(self.ckpt['numpy_random_state'])
        else:
            print("Could not find numpy random state in checkpoint.")
        if 'numpy_rng_bit_generator_state' in self.ckpt.keys():
            self.np_rng.bit_generator.state = self.ckpt['numpy_rng_bit_generator_state']
        else:
            print("Could not find numpy rng bit generator state")
        self.model.load_state_dict(self.ckpt['model_state_dict'])
        if 'optimizer_state_dict' in self.ckpt.keys():
            self.optimizer.load_state_dict(self.ckpt['optimizer_state_dict'])
        else:
            print("Could not find optimizer state in checkpoint.")
        if self.lr_scheduler:
            if 'lr_scheduler_state_dict' in self.ckpt.keys():
                self.lr_scheduler.load_state_dict(self.ckpt['lr_scheduler_state_dict'])
            else:
                print("Could not find learning rate scheduler state in checkpoint.")
        self.normalizers = self.ckpt.get('normalizers', None)

        # Load statistics
        if is_sanitized:
            return None, None, None, None
        else:
            last_epoch = sorted(self.ckpt['train_progress']['epochs'].keys())[-1]
            lowest_valid_loss = np.min(
                np.asarray(
                    [
                        self.ckpt['train_progress']['epochs'][epoch]['validation']['loss']
                        for epoch in sorted(self.ckpt['train_progress']['epochs'].keys())
                    ]
                )
            )
            training_time = self.ckpt['train_progress']['epochs'][last_epoch]['training_time_s']
            train_start_time = self.ckpt['train_progress']['start']
            return lowest_valid_loss, training_time, train_start_time, self.ckpt['train_progress']


    def _log_text(self, text: str):
        if self.tqdm_usage == TqdmUsage.SHELL:
            tqdm.write("\n")
        if self.tqdm_usage in [TqdmUsage.SHELL, TqdmUsage.NOTEBOOK]:
            tqdm.write(text)
        else:
            print(text, flush=True)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Performs combined training")
    parser.add_argument('-b', '--base-dir', type=str, required=True,
                        help="Base directory for this training run. All output"
                        " will be saved to this directory.")
    parser.add_argument('--disallow-checkpoint-loading', action='store_true',
                        help="Indicates that training should not be resumed "
                        "from a checkpoint even if one exists.")
    parser.add_argument('-v', '--verbosity', type=int, default=1,
                        help="Verbosity level for Directory Manager")
    parser.add_argument('--finetune', action="store_true",
                        help="Perform finetuning of an existing checkpoint")
    args = parser.parse_args()
    edm = ExperimentDirectoryManager(base_dir=args.base_dir,
                                     experiment_type=ExperimentType.FINETUNING if args.finetune else ExperimentType.COMBINED_TRAINING,
                                     allow_checkpoint_loading=args.finetune or not args.disallow_checkpoint_loading, # If finetuning, ALWAYS allow checkpoint loading
                                     verbosity=args.verbosity)
    config, ckpt = edm.get_modified_config_and_checkpoint()
    if args.finetune:
        trainer = CombinedTrainer(config=config, checkpoint=ckpt, finetune=True)
    elif ckpt is None:
        trainer = CombinedTrainer(config=config, checkpoint=ckpt)
    else:
        trainer = CombinedTrainer(config=None, checkpoint=ckpt)

    trainer.fit()
