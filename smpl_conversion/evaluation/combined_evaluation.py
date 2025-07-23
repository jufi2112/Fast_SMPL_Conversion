# Copyright (c) 2025 Julien Fischer, Chair of Computer Graphics and Visualization, TUD Dresden University of Technology.
# All rights reserved.
# Licensed under the MIT License. See LICENSE in the project root for license information.

import os
import yaml
import vedo
import torch
import smplx
import pickle
import argparse
import numpy as np
import torch.nn as nn

from tqdm import tqdm
from dotmap import DotMap
from os import path as osp
from time import perf_counter
from typing import List, Optional
from supr.pytorch.supr import SUPR
from tqdm.notebook import tqdm as tqdm_notebook
from smpl_conversion.losses import CorrespondenceLoss
from smpl_conversion.factories import BodyModelFactory, ConversionModelFactory
from smpl_conversion.data import SMPLConversionDataset
from smpl_conversion.utils.model_infos import MODEL_STATS
from smpl_conversion.evaluation.base_evaluator import BaseEvaluator
from smpl_conversion.data.transforms import get_data_transform_class
from smpl_conversion.utils.model_evaluation import get_model_vertices_faces
from smpl_conversion.evaluation.conversion_result_logger import ConversionResultLogger
from smpl_conversion.models import CombinedConversionNetwork, SeparatedConversionNetwork
from smpl_conversion.utils.experiment_directory_manager import ExperimentDirectoryManager
from smpl_conversion.models.combined import assemble_combined_network_input, split_network_output
from smpl_conversion.utils.enum_configurations import PoseRepresentation, ExperimentType, ConversionNetworkArchitecture, BodyModelType


class CombinedEvaluator(BaseEvaluator):
    def __init__(self,
                 checkpoint_location: str,
                 dataset_location: str,
                 body_model_location: str,
                 transfer_file_location: str,
                 device_dataset: str,
                 device_inference: str = 'auto',
                 verbosity: int = 0):
        """
            Class to evaluate a checkpoint on a dataset.

            Params
            ------
                checkpoint_location (str):
                    Path to the checkpoint that should be evaluated.
                dataset_location (str):
                    Path to the dataset '.npz' file where the checkpoint should be applied to.
                body_model_location (str):
                    Path to the body models.
                transfer_file_location (str):
                    Path to the transfer file.
                device_dataset (str):
                    Device where the whole dataset should be loaded to.
                device_inference (str):
                    Device that should be used for inference. Defaults to 'auto', i.e. automatically
                    determine whether a CUDA capable device is available and use this.
                verbosity (int):
                    Verbosity level, defaults to 0.
        """
        super().__init__(checkpoint_location, dataset_location,
                         body_model_location, transfer_file_location,
                         device_dataset, device_inference, verbosity)


    def _load_dataset(self, dataset_path):
        pose_repr = PoseRepresentation.from_string(self.config['conversion']['input_rotation_representation'])
        output_repr = PoseRepresentation.from_string(self.config['conversion']['output_rotation_representation'])
        dataset = SMPLConversionDataset(
            dataset_fpath=dataset_path,
            mode=self.config.conversion.mode,
            parameters=self.config.conversion.parameter,
            parameters_to_learn=self.config.conversion.parameter,
            gender=self.gender,
            device_loading=self.device_dataset,
            device_fetch=self.device_inference,
            rotation_representation=pose_repr,
            ground_truth_rotation_representation=output_repr,
            subsampling_factor=1,
            transform=None,
            target_transform=None,
            normalize_data=None,
            normalization_method=None,
            normalization_parameters=None,
            allow_unsupervised=True
        )
        return dataset


    def _create_model(self,
                      config: DotMap,
                      input_rotation_representation: PoseRepresentation,
                      output_rotation_representation: PoseRepresentation
                      ) -> nn.Module:
        """
            Creates a new conversion model instance

            Params
            ------
                config (DotMap):
                    The configuration of the model.
                input_rotation_representation (PoseRepresentation):
                    Required network input rotation representation
                output_rotation_representation (PoseRepresentation):
                    Network output rotation representation

            Returns
            -------
                nn.Module: The new model instance
        """
        m_from, m_to = config.conversion.mode.split('2')
        m_from = BodyModelType.from_string(m_from)
        m_to = BodyModelType.from_string(m_to)
        model_architecture = ConversionNetworkArchitecture.from_string(config.conversion.model_type)
        if model_architecture is None:
            raise ValueError(f"Unsupported value for key 'conversion.model_type': {config.conversion.model_type}")
        model = ConversionModelFactory.create_conversion_model(config, self.device_inference)
        return model


    def _predict(self,
                 X: DotMap,
                 ) -> DotMap:
        """
            Performs prediction

            Params
            ------
                X (DotMap):
                    DotMap containing the input parameters as keys 'trans', 'betas', and 'poses'

            Returns
            -------
                DotMap:
                    Predicted parameters in keys 'trans', 'betas', and 'poses'
        """
        if self.normalizer is not None:
            raise NotImplementedError("Normalization is currently not implemented.")
        network_input = assemble_combined_network_input(self.config.data.n_shape_components,
                                                        trans=X.trans,
                                                        betas=X.betas,
                                                        poses=X.poses)
        network_input = self.input_rotation_transform(network_input)
        pred = self.model(network_input)
        pred = self.output_rotation_transform(pred)
        trans, betas, poses = split_network_output(pred, self.config.data.n_shape_components)
        pred = DotMap({
            'trans': trans,
            'betas': betas,
            'poses': poses
        }, _dynamic=False)
        return pred


    def evaluate(self,
                 batch_size: int,
                 metric_names: List[str],
                 show_vertex_error_distribution: bool,
                 set_error_of_invalid_vertices_to_zero: bool,
                 tqdm_mode: str,
                 log_translation: bool,
                 shape_components_to_log: int,
                 joints_to_log: List[int],
                 result_save_location: Optional[str] = None,
                 previous_save_location: Optional[str] = None,
                 overwrite_existing_results: Optional[bool] = False
                 ):
        """
            Evaluates the checkpoint on the dataset.

            Params
            ------
                batch_size (int):
                    Batch size to use for predictions.
                metric_names (list of str):
                    Name of the metrics to calculate for evaluation purposes.
                show_vertex_error_distribution (bool):
                    Whether the distribution of the MPVPE should be visualized.
                set_error_of_invalid_vertices_to_zero (bool):
                    Whether the error of vertices that are masked out during training should be set to zero.
                tqdm_mode (str):
                    Tqdm mode. Can be 'shell', 'notebook', or 'off'. All other values will
                    evaluate to 'off'
                log_translation (bool):
                    Whether input and predicted offsets translation parameters should be logged to a file
                shape_components_to_log (int):
                    How many of the shape components should be logged to a file
                joints_to_log (list of int):
                    Which joint indices should be logged to a file
                result_save_location (str):
                    Location where result files should be written to. Defaults to None
                previous_save_location (str):
                    If a previous evaluation should be loaded, provide a path to its result file.
                    Defaults to None (i.e. do not load previous evaluation result).
                overwrite_existing_results (bool):
                    If a new result file is created, controls whether an existing file with the same name should be overwritten.
                    If false, the existing result file is loaded instead. Defaults to False.
        """
        save_dict = None
        if previous_save_location is None:
            # check that saving of the evaluation result will work, otherwise, raise an error before doing to evaluation
            path, ext = osp.splitext(result_save_location)
            if ext == '':
                # given result location is a directory, so create filename
                fname = f'evaluation_result_{self.config.conversion.mode}.pkl'
                fpath = path
                if osp.isfile(osp.join(fpath, fname)) and not overwrite_existing_results:
                    # raise ValueError(
                    #     f"A file {osp.join(fpath, fname)} already exists. Provide "
                    #     "overwrite_existing_results=True to allow overwriting of files when "
                    #     "the filename is not given and therefore automatically determined."
                    # )
                    with open(osp.join(fpath, fname), 'rb') as file:
                        save_dict = pickle.load(file)
                log_fpath = osp.join(result_save_location, 'input_prediction_log.txt')
            else:
                # given result location already contains a filename
                fname = osp.basename(result_save_location)
                fpath = osp.dirname(result_save_location)
                if osp.isfile(osp.join(fname, fpath)) and not overwrite_existing_results:
                    with open(osp.join(fpath, fname), 'rb') as file:
                        save_dict = pickle.load(file)
                log_fpath = osp.join(fpath, 'input_prediction_log.txt')

            os.makedirs(fpath, exist_ok=True)
            assert isinstance(metric_names, list), f"Expected parameter 'metric_names' to be of type list, but got type {type(metric_names)}"
            self.data_loader = self._create_data_loader(batch_size)
            metrics_fn = CorrespondenceLoss(self.model_from,
                                            self.model_to,
                                            self.gender,
                                            self.body_model_location,
                                            self.transfer_file_location,
                                            batch_size,
                                            self.config.data.n_shape_components,
                                            self.device_inference,
                                            self.config.conversion.supr_is_constrained,
                                            PoseRepresentation.from_string(self.config['conversion']['input_rotation_representation']),
                                            PoseRepresentation.from_string(self.config['conversion']['output_rotation_representation'])
                                            )
            if save_dict is None:
                if tqdm_mode == 'shell':
                    loop = tqdm
                elif tqdm_mode == 'notebook':
                    loop = tqdm_notebook
                else:
                    loop = None
                num_metrics = len(metric_names)
                dataset_size = len(self.dataset)
                total_metrics = torch.zeros((num_metrics,), dtype=torch.float32)
                n_vertices_target_model = MODEL_STATS[self.model_to].verts
                mean_per_vertex_score = torch.zeros((n_vertices_target_model), dtype=torch.float32)
                total_time_inference = 0
                mask_ids = None

                evaluation_start_time = perf_counter()
                self.model.eval()
                enable_logging = log_translation or shape_components_to_log != 0 or joints_to_log is None or (isinstance(joints_to_log, list) and len(joints_to_log) > 0)
                with ConversionResultLogger(log_fpath, self.model_from, self.model_to,
                                            PoseRepresentation.from_string(self.config['conversion']['output_rotation_representation']),
                                            self.config['data']['n_shape_components'], enable_logging, log_translation,
                                            shape_components_to_log, joints_to_log) as logger:
                    with torch.no_grad():
                        for batch_idx, (X, _) in enumerate(self.data_loader) if loop is None else enumerate(loop(self.data_loader, desc="Inference", position=0, leave=True)):

                            batch_per_vertex_loss_updated = False
                            bs = len(X.poses)
                            batch_weight = bs / dataset_size

                            input_params = DotMap({
                                'trans': X.trans,
                                'betas': X.betas[..., :self.config.data.n_shape_components],
                                'poses': X.poses
                            }, _dynamic=False)
                            inference_start_time = perf_counter()
                            pred = self._predict(X)
                            inference_time = perf_counter() - inference_start_time
                            total_time_inference += inference_time
                            logger.log(input_params, pred)

                            # calculate metrics
                            for idx, metric in enumerate(metric_names):
                                score, per_vertex_score, valid_vertices = metrics_fn(input_params,
                                                                                     pred,
                                                                                     loss_type=metric,
                                                                                     reduction_mode="mean",
                                                                                     return_per_vertex_loss=True
                                                                                     )
                                if mask_ids is None and valid_vertices is not None:
                                    mask_ids = valid_vertices
                                total_metrics[idx] += score.item() * batch_weight
                                if metric == "vertex" and not batch_per_vertex_loss_updated:
                                    batch_per_vertex_loss_updated = True
                                    mean_per_vertex_score += per_vertex_score * batch_weight
                        if not batch_per_vertex_loss_updated:
                            print(f"Warning: Total per vertex loss has not been updated in batch {batch_idx}", flush=True)
                evaluation_time = perf_counter() - evaluation_start_time

                save_dict = {
                    'mode': self.config.conversion.mode,
                    'inference_s': total_time_inference,
                    'evaluation_s': evaluation_time,
                    'dataset_size': dataset_size,
                    'batch_size': batch_size,
                    'mean_per_vertex_score': mean_per_vertex_score,
                    'dataset': self.dataset_fpath,
                    'checkpoint': self.checkpoint_fpath,
                    'mask_ids': mask_ids,
                    'metrics': {}

                }
                save_dict['metrics'] = {
                    metric: total_metrics[idx]
                    for idx, metric in enumerate(metric_names)
                }

                with open(osp.join(fpath, fname), 'wb') as file:
                    pickle.dump(save_dict, file, protocol=pickle.HIGHEST_PROTOCOL)

        else:
            metrics_fn = CorrespondenceLoss(self.model_from,
                                            self.model_to,
                                            self.gender,
                                            self.body_model_location,
                                            self.transfer_file_location,
                                            1,
                                            self.config.data.n_shape_components,
                                            self.device_inference,
                                            self.config.conversion.supr_is_constrained,
                                            PoseRepresentation.from_string(self.config['conversion']['input_rotation_representation']),
                                            PoseRepresentation.from_string(self.config['conversion']['output_rotation_representation'])
                                            )
            # load previous evaluation
            if not osp.isfile(previous_save_location):
                raise ValueError(f"The provided previous save location is not a valid file: {previous_save_location}")
            with open(previous_save_location, 'rb') as file:
                save_dict = pickle.load(file)

        if self.verbosity > 0:
            print("============================")
            print("== Evaluation Information ==")
            print("============================")
            print(f"Checkpoint: {save_dict['checkpoint']}")
            print(f"Conversion mode: {save_dict['mode']} with batch size of {save_dict['batch_size']}", flush=True)
            if self.model_from == BodyModelType.SUPR or self.model_to == BodyModelType.SUPR:
                print(f"SUPR model type: {'constrained' if self.supr_constrained else 'unconstrained'}")
            print(f"Dataset: {save_dict['dataset']} with {save_dict['dataset_size']} elements", flush=True)
            print(f"Total evaluation time : {save_dict['evaluation_s']} seconds.", flush=True)
            print(f"Total inference time: {save_dict['inference_s']} seconds", flush=True)
            metrics_txt_file = []
            print("Metrics:", flush=True)
            metrics_txt_file.append('Metrics:')
            for idx, (metric, score) in enumerate(save_dict['metrics'].items()):
                print(f"   {metric}: {score}", flush=True)
                metrics_txt_file.append(f"   {metric}: {score}")
            with open(osp.join(result_save_location, 'metric_results.txt'), 'w') as metrics_file:
                metrics_file.writelines("\n".join(metrics_txt_file))

        if show_vertex_error_distribution:
            mask_ids = save_dict['mask_ids']
            # if enabled, set the error of invalid vertices to zero
            if set_error_of_invalid_vertices_to_zero and mask_ids is not None:
                mean_per_vertex_score_vis = torch.zeros_like(save_dict['mean_per_vertex_score'])
                mean_per_vertex_score_vis[mask_ids] = (save_dict['mean_per_vertex_score'])[mask_ids]
            else:
                mean_per_vertex_score_vis = save_dict['mean_per_vertex_score']

            # Convert to cm
            mean_per_vertex_score_vis *= 100.0
            if isinstance(mean_per_vertex_score_vis, torch.Tensor):
                mean_per_vertex_score_vis = mean_per_vertex_score_vis.detach().cpu().numpy()

            body_model, _ = BodyModelFactory.create_body_model(self.model_to,
                                                               self.gender,
                                                               self.config['data']['n_shape_components'],
                                                               self.body_model_location,
                                                               self.device_inference,
                                                               False,
                                                               expected_batch_size=1,
                                                               supr_is_constrained=self.supr_constrained
                                                               )
            vertices, faces = get_model_vertices_faces(body_model,
                                                       PoseRepresentation.ROTATION_VECTOR,
                                                       {'trans': torch.zeros((1, 3), dtype=torch.float32, device=self.device_inference),
                                                        'betas': torch.zeros((1, self.config['data']['n_shape_components']), dtype=torch.float32, device=self.device_inference),
                                                        'poses': torch.zeros((1, MODEL_STATS[self.model_to].pose_params), dtype=torch.float32, device=self.device_inference)},
                                                        False,
                                                        self.device_inference
                                                        )
            vertices = vertices[0]

            # if 'batch_size' in model_params:
            #     # SMPL-X / SMPL+H / SMPL model
            #     model_params['batch_size'] = 1
            #     model = smplx.create(**model_params)
            #     vertices = model().vertices[0].detach().cpu().numpy()
            #     faces = model.faces
            # else:
            #     # SUPR
            #     model = SUPR(**model_params)
            #     vertices = model(trans=torch.zeros((1,3), device="cuda"),
            #                     betas=torch.zeros((1, self.config.data.n_shape_components), device="cuda"),
            #                     pose=torch.zeros((1, MODEL_STATS.supr.pose_params), device="cuda")
            #                     )[0].detach().cpu().numpy()
            #     faces = model.faces
            # if isinstance(faces, torch.Tensor):
            #     faces = faces.detach().cpu().numpy()
            min_vertex_error = np.min(mean_per_vertex_score_vis)
            max_vertex_error = np.max(mean_per_vertex_score_vis)
            vertex_y_min = np.min(vertices[:, 1])
            body_mesh = vedo.Mesh([vertices - np.asarray([0, vertex_y_min, 0]), faces]).cmap("jet",
                                                        mean_per_vertex_score_vis,
                                                        vmin=min_vertex_error,
                                                        vmax=max_vertex_error
                                                        )
            body_mesh.add_scalarbar3d(title="Mean Error (cm)", size=(0.1, 1))

            plotter = vedo.Plotter(offscreen=True, size=(3840, 2160))
            plotter.look_at("xy")
            plotter.show([body_mesh], title="Conversion Mean Error Visualization", axes=1)
            screenshot_path = osp.join(result_save_location, 'screenshot')
            plotter.camera.SetClippingRange(0.1, 10)

            for label in ['right', 'back', 'left', 'front']:
                body_mesh.rotate_y(90)
                plotter.screenshot(screenshot_path+f'_{label}.png')

            plotter.camera.SetPosition(0, 5, 0)
            plotter.camera.SetFocalPoint(0, 0, 0)
            plotter.camera.SetViewUp(0, 0, -1)
            body_mesh.scalarbar.rotate_x(-90)
            body_mesh.scalarbar.pos([1.5, 1, 0])
            plotter.screenshot(screenshot_path+'_top.png')
            plotter.camera.SetPosition(0, -3, 0)
            plotter.camera.SetViewUp(0,0, 1)
            body_mesh.scalarbar.rotate_x(180)
            body_mesh.scalarbar.pos([1.5, 1, 0])
            plotter.screenshot(screenshot_path+'_bottom.png')

        if self.verbosity > 0:
            print("Done")
        return save_dict


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Evaluates the given combined"
                                     " training run using the provided "
                                     "configuration file.")
    parser.add_argument('-b', '--base-dir', type=str, required=True,
                        help="Base directory of the training run that should "
                        "get evaluated.")
    parser.add_argument('-c', '--config', type=str, required=True,
                        help="Path to the configuration file that should be "
                        " used for evaluation.")
    parser.add_argument('-v', '--verbosity', type=int, default=1,
                        help="Verbosity level for Directory Manager")
    args = parser.parse_args()
    edm = ExperimentDirectoryManager(base_dir=args.base_dir,
                                     experiment_type=ExperimentType.COMBINED_EVALUATION,
                                     evaluation_config=args.config,
                                     verbosity=args.verbosity
                                     )
    config, _ = edm.get_modified_config_and_checkpoint()
    if config['evaluation']['show_vertex_error_distribution']:
        vedo.settings.default_backend = 'vtk'
        vedo.settings.multi_samples = 8
    comb_evaluator = CombinedEvaluator(config["checkpoint"],
                                       config["dataset"],
                                       config["body_model_location"],
                                       config["transfer_file_location"],
                                       config["dataset_device"],
                                       config["inference_device"],
                                       config["verbosity"]
                                       )
    comb_evaluator.evaluate(**(config['evaluation']))
