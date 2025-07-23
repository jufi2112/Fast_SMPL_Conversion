# Copyright (c) 2025 Julien Fischer, Chair of Computer Graphics and Visualization, TUD Dresden University of Technology.
# All rights reserved.
# Licensed under the MIT License. See LICENSE in the project root for license information.

import os
import copy
import torch
import pickle
import numpy as np
import matplotlib.pyplot as plt

from os import path as osp
from typing import Dict, Union, Tuple, List, Optional
from smpl_conversion.utils.misc import GridIndexTracker
from smpl_conversion.utils.enum_configurations import PoseRepresentation

class TrainingVisualizer:
    def __init__(self,
                 train_progress: Union[Dict, str],
                 prev_train_run_comparison: Union[Dict, str] = None,
                 baselines: Union[str, Dict[str, Dict[str, float]]] = None
                 ):
        """
            Class to visualize a training.

            Params
            ------
                train_progress (dict or str):
                    The training to visualize. Can be either the train progress dictionary,
                    a path to a training checkpoint, or a path to the base directory of
                    the training run. In this case, the program automatically searches
                    for the training log.
                prev_train_run_comparison (dict or str):
                    A previous training run which is visualized for comparison. Optional,
                    defaults to None. Can also just point to the base directory.
                baselines (dict or str):
                    If a constant baseline should be visualized for each metric / loss,
                    a dictionary with the metric name as key and the score as value can be
                    given. Optional, defaults to None.
        """
        if isinstance(train_progress, str):
            self.train_progress = self._load_progress_from_file(train_progress)
        else:
            self.train_progress = copy.deepcopy(train_progress)
        if isinstance(prev_train_run_comparison, str):
            self.prev_train_progress = self._load_progress_from_file(prev_train_run_comparison)
        elif isinstance(prev_train_run_comparison, dict):
            self.prev_train_progress = copy.deepcopy(prev_train_run_comparison)
        else:
            self.prev_train_progress = None
        if isinstance(baselines, str):
            with open(baselines, 'rb') as file:
                self.baselines = pickle.load(file)
            if not isinstance(baselines, dict):
                raise ValueError(
                    "The given baseline file does not contain a valid dictionary."
                )
        elif isinstance(baselines, dict):
            self.baselines = copy.deepcopy(baselines)
        else:
            self.baselines = None


    def visualize(self,
                  return_figure: bool = False,
                  line_spacing: float = 0.08,
                  auto_adapt_line_spacing: bool = True,
                  ignore_loss_in_metrics_figure: bool = False,
                  combine_metrics_into_one_figure: bool = False,
                  jointly_plot_metrics: Optional[List[Tuple[int]]] = None,
                  combine_per_model_type_into_one_figure: bool = True,
                  plot_lr: bool = True,
                  plot_information: bool = True,
                  max_cols: int = 3,
                  **kwargs
                  ) -> Union[None, Tuple[plt.Figure, List[plt.Axes]]]:
        """
            Visualizes progression of loss / metrics over training epochs.

            Params
            ------
                return_figure (bool):
                    Wether the matplotlib figure and axes objects should be returned.
                    If false, the figure will be displayed. Optional, defaults to False.
                line_spacing (float):
                    Line spacing between text rows. Optional, defaults to 0.08
                auto_adapt_line_spacing (bool):
                    Whether line spacing should automatically be inferred based on the
                    available space and the rows to be displayed. Optional, defaults to True
                ignore_loss_in_metrics_figure (bool):
                    If the loss is also included in the metrics, whether it should be
                    ignored when producing the metrics figure(s). Optional, defaults to False.
                combine_metrics_into_one_figure (bool):
                    Whether all metrics should be visualized in a single plot (single axis) or
                    not. If not, a separate plot (axis) will be created for each metric.
                    Optional, defaults to True.
                jointly_plot_metrics (List of tuple of int):
                    Not implemented
                max_cols (int):
                    Maximum number of columns (sub-plots in horizontal direction) the overall
                    figure should consist of. Optional, defaults to 3

            Returns
            -------
                If return_figure is true, a tuple that contains matplotlib's figure
                and axes objects, otherwise None.
        """
        if jointly_plot_metrics is not None:
            raise NotImplementedError("Argument jointly_plot_metrics is not implemented.")
        linestyle = kwargs.get('linestyle', 'solid')
        linestyle_prev = kwargs.get('linestyle_prev', 'dotted')
        linestyle_baseline = kwargs.get('linestyle_baseline', 'dashed')
        linestyle_gt = kwargs.get('linestyle_gt', 'dashed')
        color_train = kwargs.get('color_train', 'C0')
        color_valid = kwargs.get('color_valid', 'C1')
        loss_lower_limit = kwargs.get('loss_lower_limit', None)
        loss_upper_limit = kwargs.get('loss_upper_limit', None)
        ignore_metrics = kwargs.get('ignore_metrics', [])
        (
            loss_name, number_training_epochs, train_loss, valid_loss
        ) = self._get_train_valid_loss()

        # check if per model type loss is provided (from combined training)
        (
            model_types, train_loss_per_model_type, valid_loss_per_model_type
        ) = self._get_per_model_train_valid_loss()

        # If previous training run is provided for comparison, extract information
        (
            prev_training_epochs_to_sample, prev_train_loss, prev_valid_loss
        ) = self._get_prev_train_valid_loss(loss_name)
        (
            prev_shared_models, prev_train_loss_per_model_type,
            prev_valid_loss_per_model_type
        ) = self._get_prev_per_model_train_valid_loss(prev_training_epochs_to_sample,
                                                      model_types
                                                      )
        # if previous training run has per-model type losses, we automatically want
        # to create a separate figure for each model type
        if prev_train_loss_per_model_type is not None or prev_valid_loss_per_model_type is not None:
            combine_per_model_type_into_one_figure = False

        (
            metric_names, metric_names_train, metric_names_valid, train_metrics,
            valid_metrics
        ) = self._get_train_valid_metrics(ignore_loss_in_metrics_figure, ignore_metrics)
        metrics_limits = {
            metric_name: {
                'y_upper': kwargs.get(f'{metric_name}_upper_limit', None),
                'y_lower': kwargs.get(f'{metric_name}_lower_limit', None)
            }
            for metric_name in metric_names
        }

        (
            prev_metric_names, prev_metric_names_train, prev_metric_names_valid,
            prev_train_metrics, prev_valid_metrics
        ) = self._get_prev_train_valid_metrics(metric_names,
                                               metric_names_train,
                                               metric_names_valid,
                                               prev_training_epochs_to_sample
                                               )

        lrs = self._get_lrs()
        prev_lrs = self._get_prev_lrs(prev_training_epochs_to_sample)

        n_rows, n_cols = self._determine_fig_layout(max_cols,
                                                    metric_names_train,
                                                    metric_names_valid,
                                                    combine_metrics_into_one_figure,
                                                    model_types,
                                                    combine_per_model_type_into_one_figure,
                                                    plot_lr,
                                                    plot_information,
                                                    self.prev_train_progress is not None)

        figsize = kwargs.get('figsize', (8*n_cols, 6*n_rows))
        fig, axs = plt.subplots(n_rows, n_cols, figsize=figsize)

        idx_tracker = GridIndexTracker(n_rows, n_cols)

        axs = self._plot_loss(train_loss,
                              valid_loss,
                              loss_name,
                              axs,
                              idx_tracker,
                              color_train,
                              color_valid,
                              linestyle,
                              linestyle_prev,
                              linestyle_baseline,
                              None if self.prev_train_progress is None else prev_train_loss,
                              None if self.prev_train_progress is None else prev_valid_loss,
                              self.baselines,
                              loss_upper_limit,
                              loss_lower_limit
                              )

        if model_types is not None:
            axs = self._plot_per_model_loss(train_loss_per_model_type,
                                            valid_loss_per_model_type,
                                            loss_name,
                                            prev_train_loss_per_model_type,
                                            prev_valid_loss_per_model_type,
                                            combine_per_model_type_into_one_figure,
                                            axs,
                                            idx_tracker
                                            )

        axs = self._plot_metrics(train_metrics,
                                 valid_metrics,
                                 metric_names,
                                 metric_names_train,
                                 metric_names_valid,
                                 axs,
                                 idx_tracker,
                                 combine_metrics_into_one_figure,
                                 number_training_epochs,
                                 metrics_limits,
                                 color_train,
                                 color_valid,
                                 linestyle,
                                 linestyle_prev,
                                 linestyle_baseline,
                                 linestyle_gt,
                                 prev_train_metrics,
                                 prev_valid_metrics,
                                 self.baselines
                                 )

        if plot_lr:
            axs = self._plot_lr(lrs,
                                axs,
                                idx_tracker,
                                prev_lrs
                                )

        if plot_information:
            axs = self._plot_information(train_loss,
                                         prev_train_loss,
                                         valid_loss,
                                         prev_valid_loss,
                                         axs,
                                         idx_tracker,
                                         auto_adapt_line_spacing,
                                         line_spacing
                                         )
        # Hide unused subplots
        while (True):
            try:
                idx_tracker.next()
                axs[idx_tracker()].axis('off')
            except IndexError:
                break

        fig.suptitle(f'Training Overview')
        if return_figure:
            return fig, axs
        plt.show()


    def _load_progress_from_file(self, fpath: str) -> Dict:
        if fpath == '':
            return None
        if osp.isdir(fpath):
            # Search for training log in base dir
            log_dir = osp.join(fpath, 'train_logs')
            if not osp.isdir(log_dir):
                raise FileNotFoundError(f"Could not find train_log directory inside {fpath}")
            pkl_files = [file for file in os.listdir(log_dir) if osp.splitext(file)[1] == '.pkl']
            if len(pkl_files) == 0:
                raise FileNotFoundError(f"Could not find training log in directory {log_dir}")
            if len(pkl_files) > 1:
                print(f"Found multiple training logs in directory {log_dir}, choosing first one")
            fpath = osp.join(log_dir, pkl_files[0])
        if not osp.exists(fpath):
            raise FileNotFoundError(f"Could not find file {fpath}")
        ext = osp.splitext(fpath)[1]
        if ext == '.ckpt':
            ckpt = torch.load(fpath)
            if 'train_progress' not in ckpt.keys():
                raise ValueError(f"The provided checkpoint does not contain a key 'train_progress': {fpath}")
            prog = ckpt['train_progress']
        elif ext == '.pkl':
            with open(fpath, 'rb') as file:
                prog = pickle.load(file)
        else:
            raise ValueError(f"Unsupported file extension {ext} for file {fpath}")
        return prog


    def _plot_loss(self,
                   train_loss,
                   valid_loss,
                   loss_name,
                   axs,
                   idx_tracker,
                   color_train,
                   color_valid,
                   linestyle,
                   linestyle_prev,
                   linestyle_baseline,
                   prev_train_loss = None,
                   prev_valid_loss = None,
                   baselines=None,
                   upper_y_limit: float = None,
                   lower_y_limit: float = None
                   ):
        idx_tracker.next()
        number_epochs = len(train_loss)
        # This training's train loss
        ax = axs[idx_tracker()]
        ax.plot(np.arange(1, len(train_loss) + 1),
                train_loss,
                label='train',
                color=color_train,
                linestyle=linestyle
                )
        # Plot previous training's train loss for comparison, if given
        if prev_train_loss is not None:
            ax.plot(np.arange(1, len(prev_train_loss) + 1),
                    prev_train_loss,
                    color=color_train,
                    linestyle=linestyle_prev,
                    label='prev train'
                    )
        # Plot baseline, if given
        if baselines:
            if loss_name in list(baselines['train'].keys()):
                train_loss_baseline = np.repeat(
                    baselines['train'][loss_name],
                    number_epochs
                )
                ax.plot(np.arange(1, len(train_loss_baseline) + 1),
                        train_loss_baseline,
                        color=color_train,
                        linestyle=linestyle_baseline,
                        label='baseline train'
                        )
        # This training's validation loss
        ax.plot(np.arange(1, len(valid_loss) + 1),
                valid_loss,
                label='valid',
                color=color_valid,
                linestyle=linestyle
                )
        # Plot previous training's validation loss for comparison, if given
        if prev_valid_loss is not None:
            ax.plot(np.arange(1, len(prev_valid_loss) + 1),
                    prev_valid_loss,
                    color=color_valid,
                    linestyle=linestyle_prev,
                    label='prev valid'
                    )
        if baselines:
            if loss_name in list(baselines['valid'].keys()):
                valid_loss_baseline = np.repeat(
                    baselines['valid'][loss_name],
                    number_epochs
                )
                ax.plot(np.arange(1, len(valid_loss_baseline) + 1),
                        valid_loss_baseline,
                        color=color_valid,
                        linestyle=linestyle_baseline,
                        label='baseline valid'
                        )
        ax.set_xlabel('Epochs')
        ax.set_ylabel('Loss')
        ax.set_xlim(xmin=1)
        ax.set_ylim(bottom=lower_y_limit, top=upper_y_limit)
        ax.set_title(f'Train and Validation {loss_name} Loss')
        ax.legend()
        ax.grid()
        axs[idx_tracker()] = ax
        return axs


    def _plot_per_model_loss(self,
                             per_model_train_loss,
                             per_model_valid_loss,
                             loss_name,
                             prev_per_model_train_loss,
                             prev_per_model_valid_loss,
                             combine_into_one_figure,
                             axs,
                             idx_tracker
                             ):
        models = list(per_model_train_loss.keys())
        number_epochs = len(per_model_train_loss[models[0]])
        if combine_into_one_figure:
            idx_tracker.next()
            ax = axs[idx_tracker()]
            #linestyles = ['solid', 'dotted', 'dashed', 'dashdot']
            linestyle_train = 'solid'
            linestyle_valid = 'dotted'
            # Plot training losses
            model_type_colors = []
            for n, model in enumerate(models):
                ax.plot(np.arange(1, number_epochs + 1),
                        per_model_train_loss[model],
                        label=f"{model.upper()} train",
                        linestyle=linestyle_train
                        )
                model_type_colors.append(ax.get_lines()[-1].get_color())
            # Plot validation losses
            for n, model in enumerate(models):
                ax.plot(np.arange(1, number_epochs + 1),
                        per_model_valid_loss[model],
                        label=f"{model.upper()} valid",
                        color=model_type_colors[n],
                        linestyle=linestyle_valid
                        )
            ax.set_xlabel('Epochs')
            ax.set_ylabel('Loss')
            ax.set_xlim(xmin=1)
            ax.set_title(f'Per-Model Train and Validation {loss_name} Loss')
            ax.legend()
            ax.grid()
            axs[idx_tracker()] = ax
        else:
            linestyle_current = 'solid'
            linestyle_previous = 'dotted'
            for model_type in models:
                idx_tracker.next()
                ax = axs[idx_tracker()]
                ax.plot(np.arange(1, number_epochs + 1),
                        per_model_train_loss[model_type],
                        label="train",
                        linestyle=linestyle_current
                        )
                color_train = ax.get_lines()[-1].get_color()
                if model_type in prev_per_model_train_loss.keys():
                    ax.plot(np.arange(1, number_epochs + 1),
                            prev_per_model_train_loss[model_type],
                            label="prev train",
                            linestyle=linestyle_previous,
                            color=color_train
                            )
                ax.plot(np.arange(1, number_epochs + 1),
                        per_model_valid_loss[model_type],
                        label="valid",
                        linestyle=linestyle_current
                        )
                color_valid = ax.get_lines()[-1].get_color()
                if model_type in prev_per_model_valid_loss.keys():
                    ax.plot(np.arange(1, number_epochs + 1),
                            prev_per_model_valid_loss[model_type],
                            label="prev valid",
                            linestyle=linestyle_previous,
                            color=color_valid)
                ax.set_xlabel('Epochs')
                ax.set_ylabel('Loss')
                ax.set_xlim(xmin=1)
                ax.set_title(f'{model_type.upper()} Train and Validation {loss_name} Loss')
                ax.legend()
                ax.grid()
                axs[idx_tracker()] = ax
        return axs


    def _plot_lr(self,
                 lrs,
                 axs,
                 idx_tracker,
                 prev_lrs=None
                 ):
        linestyle_prev = 'dotted'
        idx_tracker.next()
        ax = axs[idx_tracker()]
        ax.plot(np.arange(1, len(lrs) + 1), lrs, label='learning rate')
        if prev_lrs is not None:
            color = ax.get_lines()[0].get_color()
            ax.plot(np.arange(1, len(prev_lrs) + 1),
                    prev_lrs,
                    color=color,
                    linestyle=linestyle_prev,
                    label=f'prev learning rate'
                    )
        ax.set_xlabel('Epochs')
        ax.set_ylabel('Learning Rate')
        ax.set_xlim(xmin=1)
        ax.set_title('Learning Rate Scheduling')
        if prev_lrs is not None:
            ax.legend()
        ax.grid()
        axs[idx_tracker()] = ax
        return axs


    def _plot_metrics(self,
                      train_metrics,
                      valid_metrics,
                      metric_names,
                      metric_names_train,
                      metric_names_valid,
                      axs,
                      idx_tracker,
                      combine_into_one_figure: bool,
                      number_training_epochs: int,
                      metric_limits,
                      color_train,
                      color_valid,
                      linestyle,
                      linestyle_prev,
                      linestyle_baseline,
                      linestyle_gt,
                      prev_train_metrics=None,
                      prev_valid_metrics=None,
                      baselines=None,
                      ):
        for idx, metric in enumerate(metric_names):
            if metric.split('_')[0] == 'gt':
                continue
            if idx == 0 or not combine_into_one_figure:
                idx_tracker.next()
                ax = axs[idx_tracker()]
            # Plot current training's metric
            if metric in metric_names_train:
                ax.plot(np.arange(1, len(train_metrics[metric]) + 1),
                        train_metrics[metric],
                        linestyle=linestyle,
                        color=color_train,
                        label=f"train{' ' + metric if combine_into_one_figure else ''}"
                        )
                # Plot previous training's train metric, if applicable
                if prev_train_metrics is not None:
                    if metric in list(prev_train_metrics.keys()):
                        ax.plot(np.arange(1, len(prev_train_metrics[metric]) + 1),
                                prev_train_metrics[metric],
                                color=color_train,
                                linestyle=linestyle_prev,
                                label=f"prev train{' ' + metric if combine_into_one_figure else ''}"
                                )

                ### !!!EXPERIMENTAL!!!
                # Plot AMASS baseline
                gt_name = f"gt_{metric}"
                if gt_name in metric_names_train:
                    ax.plot(np.arange(1, len(train_metrics[gt_name]) + 1),
                            train_metrics[gt_name],
                            color=color_train,
                            linestyle=linestyle_gt,
                            label="AMASS baseline")
                else:
                    # GT not available
                    print(f"Found no training ground truth for metric: {metric}")
                ###

                # Plot training baseline, if given
                if baselines:
                    if metric in list(baselines['train'].keys()):
                        values = np.repeat(baselines['train'][metric], number_training_epochs)
                        ax.plot(np.arange(1, len(values) + 1),
                                values,
                                color=color_train,
                                linestyle=linestyle_baseline,
                                label=f"baseline train{' ' + metric if combine_into_one_figure else ''}"
                                )
            if metric in metric_names_valid:
                # Plot current training's validation metric
                ax.plot(np.arange(1, len(valid_metrics[metric]) + 1),
                        valid_metrics[metric],
                        linestyle=linestyle,
                        color=color_valid,
                        label=f"valid{' ' + metric if combine_into_one_figure else ''}"
                        )
                # Plot previous training's validation metric, if applicable
                if prev_valid_metrics is not None:
                    if metric in list(prev_valid_metrics.keys()):
                        ax.plot(np.arange(1, len(prev_valid_metrics[metric]) + 1),
                                prev_valid_metrics[metric],
                                color=color_valid,
                                linestyle=linestyle_prev,
                                label=f"prev valid{' ' + metric if combine_into_one_figure else ''}"
                                )

                ### !!!EXPERIMENTAL!!!
                # Plot AMASS baseline
                gt_name = f"gt_{metric}"
                if gt_name in metric_names_valid:
                    ax.plot(np.arange(1, len(valid_metrics[gt_name]) + 1),
                            valid_metrics[gt_name],
                            color=color_valid,
                            linestyle=linestyle_gt,
                            label="AMASS baseline")
                else:
                    # GT not available
                    print(f"Found no validation ground truth for metric: {metric}")
                ###

                # Plot validation metric baseline, if given
                if baselines:
                    if metric in list(baselines['valid'].keys()):
                        values = np.repeat(baselines['valid'][metric], number_training_epochs)
                        ax.plot(np.arange(1, len(values) + 1),
                                values,
                                color=color_valid,
                                linestyle=linestyle_baseline,
                                label=f"baseline valid{' ' + metric if combine_into_one_figure else ''}"
                                )
            if not combine_into_one_figure:
                ax.set_xlabel('Epochs')
                ax.set_ylabel('Metric Score')
                ax.set_xlim(xmin=1)
                ax.set_ylim(bottom=metric_limits[metric]['y_lower'], top=metric_limits[metric]['y_upper'])
                ax.set_title(f'Train and Validation {metric} Metric')
                ax.legend()
                ax.grid()
                axs[idx_tracker()] = ax
        if combine_into_one_figure:
            ax.set_xlabel('Epochs')
            ax.set_ylabel('Metric Score')
            ax.set_xlim(xmin=1)
            lower_limit = min([metric_limits[m]['y_lower'] for m in metric_names])
            upper_limit = max([metric_limits[m]['y_upper'] for m in metric_names])
            ax.set_ylim(bottom=lower_limit, top=upper_limit)
            ax.set_title('Train and Validation Metrics')
            ax.legend()
            ax.grid()
            axs[idx_tracker()] = ax
        return axs


    def _plot_information(self,
                          train_loss,
                          prev_train_loss,
                          valid_loss,
                          prev_valid_loss,
                          axs,
                          idx_tracker,
                          auto_adapt_line_spacing,
                          line_spacing
                          ):
        (
            best_train_epoch, best_valid_epoch, prev_best_train_epoch,
            prev_best_valid_epoch
        ) = self._get_epochs_of_best_train_valid_losses(train_loss,
                                                        valid_loss,
                                                        prev_train_loss,
                                                        prev_valid_loss
                                                        )
        for idx, progress in enumerate([self.train_progress, self.prev_train_progress]):
            if progress is None:
                continue
            try:
                idx_tracker.next()
                ax = axs[idx_tracker()]
                ax.axis('off')

                info_to_display = []
                info_to_display.append(f"Conversion mode: {progress['config']['conversion']['mode']}")
                if 'model_max_dims' in progress['config']['conversion'].keys():
                    max_dims = progress['config']['conversion']['model_max_dims']
                else:
                    max_dims = None
                info_to_display.append(f"Model type: {progress['config']['conversion']['model_type']}{f' of max dimension {max_dims}' if max_dims is not None else ''}")
                info_to_display.append(f"Data normalization: {'Yes' if progress['config']['data']['normalize_data'] else 'No'}")
                info_to_display.append(f"Input data repr.: {str(PoseRepresentation.from_string(progress['config']['conversion'].get('input_rotation_representation', 'rot_vec')))}")
                info_to_display.append(f"Output data repr.: {str(PoseRepresentation.from_string(progress['config']['conversion'].get('output_rotation_representation', 'rot_vec')))}")
                for key in list(progress.keys()):
                    if key in ['epochs', 'config', 'model_architecture', 'train_indices', 'valid_indices']:
                        continue
                    info_to_display.append(f"{key}: {progress[key]}")
                n_training_epochs = len(list(progress['epochs'].keys()))
                training_time_s = progress['epochs'][list(progress['epochs'].keys())[-1]]['training_time_s']
                hours, remainder = divmod(training_time_s, 3600)
                minutes, seconds = divmod(remainder, 60)
                info_to_display.append('Training time: {:02} HH : {:02} mm : {:02} ss'.format(int(hours), int(minutes), int(seconds)))
                info_to_display.append(f"Training epochs: {n_training_epochs}")
                info_to_display.append(f"Batch size: {progress['config']['training']['batch_size']}")
                if idx == 0:
                    info_to_display.append(f"Lowest training loss at epoch {best_train_epoch+1}: {train_loss[best_train_epoch]:>7f}")
                    info_to_display.append(f"Lowest validation loss at epoch {best_valid_epoch+1}: {valid_loss[best_valid_epoch]:>7f}")
                    if self.prev_train_progress is None:
                        title = "Training Hyperparameters"
                    else:
                        title = "Current Training's Hyperparameters"
                else:
                    if prev_best_train_epoch is not None:
                        info_to_display.append(f"Lowest training loss at epoch {prev_best_train_epoch+1}: {prev_train_loss[prev_best_train_epoch]:>7f}")
                    if prev_best_valid_epoch is not None:
                        info_to_display.append(f"Lowest validation loss at epoch {prev_best_valid_epoch+1}: {prev_valid_loss[prev_best_valid_epoch]:>7f}")
                    title = "Previous Training's Hyperparameters"
                if 'latent_space_size' in progress['config']['training'].keys():
                    info_to_display.append(f"Latent space size: {progress['config']['training']['latent_space_size']}")
            except Exception as e:
                info_to_display = ["Error while loading hyperparameters:", str(e)]

            first_elem_y = 0.5
            if len(info_to_display) > 0:
                first_elem_y += (len(info_to_display) - 1) * 0.5 * line_spacing

            if auto_adapt_line_spacing:
                while (first_elem_y > 1.0):
                    line_spacing *= 0.9
                    first_elem_y = 0.5 + (len(info_to_display) - 1) * 0.5 * line_spacing

            for idx, message in enumerate(info_to_display):
                ax.text(0, first_elem_y - idx * line_spacing, message)
            ax.set_title(title)
            axs[idx_tracker()] = ax
        return axs


    def _get_train_valid_loss(self) -> Tuple[str, int, np.ndarray, np.ndarray]:
        """Returns loss name, number of training epochs, train loss per epoch, validation loss per epoch"""
        loss_name = self.train_progress['loss_fn']
        number_epochs = len(list(self.train_progress['epochs'].keys()))
        train_loss = np.asarray(
            [
                self.train_progress['epochs'][epoch]['train']['loss']
                for epoch in sorted(self.train_progress['epochs'].keys())
            ]
        )
        valid_loss = np.asarray(
            [
                self.train_progress['epochs'][epoch]['validation']['loss']
                for epoch in sorted(self.train_progress['epochs'].keys())
            ]
        )
        return loss_name, number_epochs, train_loss, valid_loss


    def _get_per_model_train_valid_loss(self) -> Tuple[List[str], np.ndarray, np.ndarray]:
        """Returns list of model types, per model type train loss, per model type validation loss"""
        first_epoch = list(self.train_progress['epochs'].keys())[0]
        if 'loss_per_model_type' in self.train_progress['epochs'][first_epoch]['train']:
            model_types = list(self.train_progress['epochs'][first_epoch]['train']['loss_per_model_type'].keys())
            train_loss_per_model_type = {
                model_type: np.asarray(
                    [
                        self.train_progress['epochs'][epoch]['train']['loss_per_model_type'][model_type]
                        for epoch in sorted(self.train_progress['epochs'].keys())
                    ]
                )
                for model_type in model_types
            }
            valid_loss_per_model_type = {
                model_type: np.asarray(
                    [
                        self.train_progress['epochs'][epoch]['validation']['loss_per_model_type'][model_type]
                        for epoch in sorted(self.train_progress['epochs'].keys())
                    ]
                )
                for model_type in model_types
            }
        else:
            train_loss_per_model_type = None
            valid_loss_per_model_type = None
            model_types = None
        return model_types, train_loss_per_model_type, valid_loss_per_model_type


    def _get_prev_train_valid_loss(self,
                                   current_loss_name
                                   ) -> Tuple[int, np.ndarray, np.ndarray]:
        """Returns number of epochs to sample from previous training, previous train loss, previous validation loss"""
        if self.prev_train_progress is not None:
            prev_training_epochs_to_sample = min(
                len(self.train_progress['epochs']),
                len(self.prev_train_progress['epochs'])
            )
            if self.prev_train_progress['loss_fn'] != current_loss_name:
                    print(
                        "Losses differ between the given train progress and the given "
                        "previous train progress. Ignoring previous training's loss."
                    )
                    prev_train_loss = None
                    prev_valid_loss = None
            else:
                prev_train_loss = np.asarray(
                    [
                        self.prev_train_progress['epochs'][epoch]['train']['loss']
                        for epoch in range(1, prev_training_epochs_to_sample + 1)
                    ]
                )
                prev_valid_loss = np.asarray(
                    [
                        self.prev_train_progress['epochs'][epoch]['validation']['loss']
                        for epoch in range(1, prev_training_epochs_to_sample + 1)
                    ]
                )
        else:
            prev_training_epochs_to_sample = None
            prev_train_loss = None
            prev_valid_loss = None
        return prev_training_epochs_to_sample, prev_train_loss, prev_valid_loss


    def _get_prev_per_model_train_valid_loss(self,
                                             prev_training_epochs_to_sample,
                                             current_training_model_types
                                             ) -> Tuple[List[str], np.ndarray, np.ndarray]:
        """For the provided previous training run, returns list of model types shared with the current training run, per
        model type train loss, and per model type validation loss of previous run"""
        if self.prev_train_progress is not None:
            first_epoch = list(self.prev_train_progress['epochs'].keys())[0]
            if 'loss_per_model_type' in self.prev_train_progress['epochs'][first_epoch]['train']:
                prev_model_types = list(self.prev_train_progress['epochs'][first_epoch]['train']['loss_per_model_type'].keys())
                shared_model_types = [model_type for model_type in prev_model_types if model_type in current_training_model_types]
                prev_train_loss_per_model_type = {
                    model_type: np.asarray(
                        [
                            self.prev_train_progress['epochs'][epoch]['train']['loss_per_model_type'][model_type]
                            for epoch in range(1, prev_training_epochs_to_sample+1)
                        ]
                    )
                    for model_type in shared_model_types
                }
                prev_valid_loss_per_model_type = {
                    model_type: np.asarray(
                        [
                            self.prev_train_progress['epochs'][epoch]['validation']['loss_per_model_type'][model_type]
                            for epoch in range(1, prev_training_epochs_to_sample+1)
                        ]
                    )
                    for model_type in shared_model_types
                }
            else:
                prev_train_loss_per_model_type = None
                prev_valid_loss_per_model_type = None
                shared_model_types = None
        else:
            prev_train_loss_per_model_type = None
            prev_valid_loss_per_model_type = None
            shared_model_types = None
        return shared_model_types, prev_train_loss_per_model_type, prev_valid_loss_per_model_type


    def _get_train_valid_metrics(self,
                                 ignore_loss_in_metrics_figure: bool,
                                 ignore_metrics: List[str]
                                 ) -> Tuple[List[str], List[str], List[str], Dict[str, np.ndarray], Dict[str, np.ndarray]]:
        """Returns list of metric names, train metrics, validation metrics"""
        metric_names_train = list(
            self.train_progress['epochs'][list(self.train_progress['epochs'].keys())[0]]['train']['metrics'].keys()
        )
        metric_names_valid = list(
            self.train_progress['epochs'][list(self.train_progress['epochs'].keys())[0]]['validation']['metrics'].keys()
        )
        metric_names = list(set(metric_names_train + metric_names_valid))

        if ignore_loss_in_metrics_figure:
            try:
                metric_names.remove(self.train_progress['loss_fn'])
                metric_names_train.remove(self.train_progress['loss_fn'])
                metric_names_valid.remove(self.train_progress['loss_fn'])
            except ValueError:
                pass

        metric_names = [name for name in metric_names if name not in ignore_metrics]
        metric_names_train = [name for name in metric_names_train if name not in ignore_metrics]
        metric_names_valid = [name for name in metric_names_valid if name not in ignore_metrics]

        train_metrics = {
            name: np.asarray(
                [
                    self.train_progress['epochs'][epoch]['train']['metrics'][name]
                    for epoch in sorted(self.train_progress['epochs'].keys())
                ]
            )
            for name in metric_names_train
        }
        valid_metrics = {
            name: np.asarray(
                [
                    self.train_progress['epochs'][epoch]['validation']['metrics'][name]
                    for epoch in sorted(self.train_progress['epochs'].keys())
                ]
            )
            for name in metric_names_valid
        }
        return metric_names, metric_names_train, metric_names_valid, train_metrics, valid_metrics


    def _get_prev_train_valid_metrics(self,
                                      metric_names,
                                      metric_names_train,
                                      metric_names_valid,
                                      prev_training_epochs_to_sample
                                      ) -> Tuple[List[str], Dict[str, np.ndarray], Dict[str, np.ndarray]]:
        """Returns the metric names of the previous training run which are also present in the current training run,
        the previous training's train metrics, the previous training's validation metrics"""
        if self.prev_train_progress is not None:
            prev_all_metric_names_train = list(
                self.prev_train_progress['epochs'][list(self.prev_train_progress['epochs'].keys())[0]]['train']['metrics'].keys()
            )
            prev_all_metric_names_valid = list(
                self.prev_train_progress['epochs'][list(self.prev_train_progress['epochs'].keys())[0]]['validation']['metrics'].keys()
            )
            prev_metric_names_train = [name for name in prev_all_metric_names_train if name in metric_names_train]
            prev_metric_names_valid = [name for name in prev_all_metric_names_valid if name in metric_names_valid]
            prev_metric_names = list(set(prev_metric_names_train + prev_metric_names_valid))
            prev_train_metrics = {
                name: np.asarray(
                    [
                        self.prev_train_progress['epochs'][epoch]['train']['metrics'][name]
                        for epoch in range(1, prev_training_epochs_to_sample + 1)
                    ]
                )
                for name in prev_metric_names_train
            }
            prev_valid_metrics = {
                name: np.asarray(
                    [
                        self.prev_train_progress['epochs'][epoch]['validation']['metrics'][name]
                        for epoch in range(1, prev_training_epochs_to_sample + 1)
                    ]
                )
                for name in prev_metric_names_valid
            }
        else:
            prev_metric_names = None
            prev_metric_names_train = None
            prev_metric_names_valid = None
            prev_train_metrics = None
            prev_valid_metrics = None
        return prev_metric_names, prev_metric_names_train, prev_metric_names_valid, prev_train_metrics, prev_valid_metrics


    def _get_lrs(self) -> np.ndarray:
        """Returns the learning rates"""
        lrs = np.asarray(
            [
                self.train_progress['epochs'][epoch]['lr']
                for epoch in sorted(self.train_progress['epochs'].keys())
            ]
        )
        return lrs


    def _get_prev_lrs(self,
                      prev_training_epochs_to_sample
                      ) -> np.ndarray:
        """Returns the previous training's learning rate scheduling"""
        if self.prev_train_progress is not None:
            prev_lrs = np.asarray(
                [
                    self.prev_train_progress['epochs'][epoch]['lr']
                    for epoch in range(1, prev_training_epochs_to_sample + 1)
                ]
            )
        else:
            prev_lrs = None
        return prev_lrs


    def _get_epochs_of_best_train_valid_losses(self,
                                               train_loss,
                                               valid_loss,
                                               prev_train_loss,
                                               prev_valid_loss
                                               ) -> Tuple[int, int, int, int]:
        """Returns epochs at which the lowest train loss, validation loss, previous training's train loss, and previous training's validation loss where observed"""
        best_train = np.argmin(train_loss)
        best_valid = np.argmin(valid_loss)
        if prev_train_loss is not None:
            prev_best_train = np.argmin(prev_train_loss)
        else:
            prev_best_train = None
        if prev_valid_loss is not None:
            prev_best_valid = np.argmin(prev_valid_loss)
        else:
            prev_best_valid = None
        return best_train, best_valid, prev_best_train, prev_best_valid


    def _determine_fig_layout(self,
                              max_cols: int,
                              metric_names_train: List[str],
                              metric_names_valid: List[str],
                              combine_metrics_into_one_figure: bool,
                              model_types: List[str],
                              combine_per_model_type_into_one_figure: bool,
                              plot_lr: bool,
                              plot_information: bool,
                              prev_train_is_available: bool
                              ) -> Tuple[int, int]:
        """Returns number of rows and number of columns that the final figure should have"""
        # Previous training's results will be written to the same plot as the corresponding current training's results
        n_plots = 1     # train and validation loss will always be given
        if plot_lr:
            n_plots += 1
        if plot_information:
            n_plots += 1
            if prev_train_is_available:
                n_plots += 1
        # Metrics
        if metric_names_train is not None:
            if combine_metrics_into_one_figure:
                n_plots += 1
            else:
                num_metrics_plots = len(metric_names_train)
                for metric in metric_names_train:
                    if metric.startswith('gt_'):
                        if metric[3:] in metric_names_train:
                            num_metrics_plots -= 1
                n_plots += num_metrics_plots
        if metric_names_valid is not None:
            if combine_metrics_into_one_figure:
                if metric_names_train is None:
                    n_plots += 1
            else:
                num_metrics_plots = len(metric_names_train)
                for metric in metric_names_train:
                    if metric.startswith('gt_'):
                        if metric[3:] in metric_names_train:
                            num_metrics_plots -= 1
                n_plots += num_metrics_plots
        # Per model type
        if model_types is not None:
            if combine_per_model_type_into_one_figure:
                n_plots += 1
            else:
                n_plots += len(model_types)
        n_cols = min(max_cols, n_plots)
        n_rows = int(np.ceil(n_plots / n_cols))
        return n_rows, n_cols
