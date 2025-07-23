# Copyright (c) 2025 Julien Fischer, Chair of Computer Graphics and Visualization, TUD Dresden University of Technology.
# All rights reserved.
# Licensed under the MIT License. See LICENSE in the project root for license information.

import yaml
import torch
import pickle
import argparse
import numpy as np

from tqdm import tqdm
from dotmap import DotMap
from os import path as osp
from time import perf_counter
from npy_append_array import NpyAppendArray
from tqdm.notebook import tqdm as tqdm_notebook
from smpl_conversion.losses import CorrespondenceLoss
from typing import Union, Optional, List, Tuple, Dict
from smpl_conversion.inference.base_predictor import BasePredictor
from smpl_conversion.data import CombinedInferenceDataset
from smpl_conversion.models import CombinedConversionNetwork, SeparatedConversionNetwork
from smpl_conversion.models.combined import assemble_combined_network_input, split_network_output
from smpl_conversion.utils.enum_configurations import PoseRepresentation, TqdmUsage, ConversionNetworkArchitecture, BodyModelType
from smpl_conversion.factories import ConversionModelFactory


class CombinedPredictor(BasePredictor):
    def __init__(self,
                 checkpoint_location: str,
                 support_conversion_error_calculation: bool = True,
                 device: str = 'auto',
                 verbosity: int = 0,
                 body_model_location: str = None,
                 transfer_file_location: str = None,
                 expected_batch_size: int = 1
                 ):
        """
            Class for the combined prediction of translation, shape, and pose
            parameters.

            Params
            ------
                checkpoint_location (str):
                    Path to the checkpoint that should be loaded
                support_conversion_error_calculation (bool):
                    Whether inference-time calculations of conversion errors should
                    be supported. If True, users can select in the predict
                    function whether such errors should be calculated or not for
                    each run separately. Defaults to True.
                device (str):
                    Device where prediction and possibly metric calculations
                    should be performed on. Defaults to 'auto', i.e. CUDA if
                    torch.cuda.is_available() is true, otherwise CPU
                verbosity (int):
                    Verbosity level, defaults to 0.
                body_model_location (str):
                    If support_accuracy_calculation is true, the path to where the
                    body models are located. Defaults to None
                transfer_file_location (str):
                    If support_accuracy_calculation is true, the path to where the
                    transfer files are located. Defaults to None
                expected_batch_size (int):
                    If conversion error is to be calculated, the expected batch
                    size. Defaults to 1
        """
        super().__init__(checkpoint_location,
                         device,
                         verbosity)
        if support_conversion_error_calculation:
            self.metric_fn = CorrespondenceLoss(self.model_from,
                                                self.model_to,
                                                self.gender,
                                                body_model_location,
                                                transfer_file_location,
                                                expected_batch_size,
                                                self.get_number_shape_components(),
                                                self.device,
                                                verbosity=self.verbosity,
                                                supr_constrained=self.supr_constrained,
                                                input_rotation_representation=PoseRepresentation.from_string(self.config['conversion']['input_rotation_representation']),
                                                prediction_rotation_representation=PoseRepresentation.from_string(self.config['conversion']['output_rotation_representation'])
                                                )
        else:
            self.metric_fn = None


    def _create_model(self,
                      config: DotMap):
        model = ConversionModelFactory.create_conversion_model(config, self.device)
        return model


    def predict(self,
                X: Union[torch.Tensor, DotMap, np.ndarray],
                split_output: bool = False,
                errors_to_calculate: Optional[Union[List[str], str]] = None
                ) -> Tuple[Union[torch.Tensor, DotMap], Union[None, torch.Tensor]]:
        """
            Predicts the given batch of data.

            Params
            ------
                X (torch.Tensor or DotMap or np.ndarray):
                    The batch of data that should be predicted
                split_output (bool):
                    Whether the output should be split into a DotMap with keys
                    for translation, betas, and poses. Defaults to False, i.e.
                    return a single Tensor containing all data horizontally
                    stacked.
                errors_to_calculate (None or str or list of str):
                    The conversion errors that should be calculated for this batch.
                    Calculation can be disabled by providing None. If a string or a
                    list of strings is given, these errors are calculated using
                    the CorrespondenceLoss class. If error calculations have been
                    disabled in the constructor, providing an argument different
                    from None results in a ValueError. Defaults to None.
                    Possible values are: 'mpvpe', 'pvpe', 'edge'

            Returns:
                Union[torch.Tensor, DotMap]:
                    If split_output is False, a tensor with the converted
                    parameters stacked horizontally. Otherwise a DotMap with the
                    individual parameters.
                Union[None or torch.Tensor]:
                    If metrics are to be calculated, the metric(s) in the same
                    order as given in the argument. None if no metrics are to be
                    calculated.
        """
        if errors_to_calculate is not None and self.metric_fn is None:
            raise ValueError("Metric calculation has been disabled in the "
                             "constructor but metrics_to_calculate is not None!")
        translation_key = 'trans'
        if isinstance(X, np.ndarray):
            X = torch.tensor(X, dtype=torch.float32)
        elif isinstance(X, DotMap):
            if 'trans' not in X.keys() and 'transl' in X.keys():
                translation_key = 'transl'
            if isinstance(X[translation_key], np.ndarray):
                X[translation_key] = torch.tensor(X[translation_key], dtype=torch.float32)
            if X[translation_key].ndim == 1:
                X[translation_key] = X[translation_key].unsqueeze(dim=0)
            if isinstance(X.betas, np.ndarray):
                X.betas = torch.tensor(X.betas, dtype=torch.float32)
            if X.betas.ndim == 1:
                X.betas = X.betas.unsqueeze(dim=0)
            if isinstance(X.poses, np.ndarray):
                X.poses = torch.tensor(X.poses, dtype=torch.float32)
            if X.poses.ndim == 1:
                X.poses = X.poses.unsqueeze(dim=0)
            if self.verbosity > 0 and X.betas.shape[1] != self.config.data.n_shape_components:
                print(
                    f"The given checkpoint was trained with {self.config.data.n_shape_components} "
                    f"shape components, you have {X.betas.shape[1]}."
                )
            X = assemble_combined_network_input(self.config.data.n_shape_components,
                                                input_dict=X)
        if X.ndim == 1:
            X.unsqueeze_(0)
            single_dim = True
        else:
            single_dim = False
        X = X.to(self.device)
        with torch.no_grad():
            if self.normalizer is not None:
                raise NotImplementedError("Normalization is currently not supported")
                X = self.normalizer.transform(X)
            self.model.eval()
            pred = self.model(X)
            if self.normalizer is not None:
                pred = self.normalizer.inverse_transform(pred)
        # Calculate metrics
        if errors_to_calculate is not None:
            if isinstance(errors_to_calculate, str):
                single_error = True
                errors_to_calculate = [errors_to_calculate]
            else:
                single_error = False
            errors = []
            for metric in errors_to_calculate:
                if metric.lower() == 'mpvpe':
                    loss_type = 'vertex'
                    red_mode = 'mean'
                    pvpe = False
                elif metric.lower() == 'mpvpe_pvpe':
                    loss_type = 'vertex'
                    red_mode = 'mean'
                    pvpe = True
                elif metric.lower() == 'edge':
                    loss_type = 'edge'
                    red_mode = 'mean'
                    pvpe = False
                else:
                    print(f"Unsupported error type: {metric}, skipping...")
                    errors.append(-1.)
                    continue
                with torch.no_grad():
                    error = self.metric_fn(X, pred, loss_type, red_mode, pvpe)
                    # TODO: right now, PVPE is tossed away, save it somehow
                    if not pvpe:
                        errors.append(error.item())
                    else:
                        errors.append(error[0].item())
            if single_error:
                errors = errors[0]
            errors = torch.tensor(errors, dtype=torch.float32)
        else:
            errors = None

        if split_output:
            trans, betas, poses = split_network_output(pred, self.config.data.n_shape_components)
            pred = DotMap({
                translation_key: trans[0] if single_dim else trans,
                'betas': betas[0] if single_dim else betas,
                'poses': poses[0] if single_dim else poses
            }, _dynamic=False)
        else:
            if single_dim:
                pred = pred[0]
        return pred, errors


    def predict_dataset(self,
                        input_dataset: str,
                        output_file: Union[str, None],
                        return_results: bool,
                        translation_key: str = None,
                        shape_key: str = None,
                        pose_key: str = None,
                        batch_size: int = 1,
                        subsampling_factor: int = 1,
                        tqdm_usage: TqdmUsage = TqdmUsage.OFF,
                        errors_to_calculate: Union[None, List[str], str] = None
                        ) -> Tuple[Union[None, torch.Tensor], Dict]:
        """
            Performs parameter conversion on whole dataset.

            Params
            ------
                input_dataset (str):
                    Dataset that should be converted. If the input is a .npy file,
                    it's expected to contain all parameters horizontally stacked
                    like this: [translation, shape, pose]. If this is not the case,
                    provide the translation_key, shape_key, and pose_key parameters
                output_file (str or None):
                    File to which the output should be written. If None, the
                    conversion results will not be written to file.
                return_results (bool):
                    Whether the conversion results should be returned after the
                    prediction finished.
                translation_key (str):
                    Key under which the translation parameters are stored in the
                    dataset. Defaults to None
                shape_key (str):
                    Key under which the shape parameters are stored in the dataset.
                    Defaults to None
                pose_key (str):
                    Key under which the pose parameters are stored in the dataset.
                    Defaults to None
                batch_size (int):
                    Batch size to convert the dataset with. Defaults to 1
                errors_to_calculate (None or list of str or str):
                    The conversion errors that should be calculated for the dataset.
                    Calculation can be disabled by providing None. If a string or a
                    list of strings is given, these errors are calculated using
                    the CorrespondenceLoss class. If error calculations have been
                    disabled in the constructor, providing an argument different
                    from None results in a ValueError. Defaults to None.
                    Possible values are: 'mpvpe', 'pvpe', 'edge'

            Returns
            -------
                None or torch.Tensor:
                    If return_results is True the conversion result, None otherwise
                Dict:
                    Conversion conversion metrics, if no errors were to be
                    calculated, this dict only contains one entry ('runtime_s')
                    for the conversion runtime.
        """
        if output_file is not None:
            if osp.isdir(output_file):
                osp.join(output_file, 'conversion_results.npy')
            if osp.isfile(output_file):
                raise ValueError(f"The provided output file already exists: {output_file}")

        dataset = CombinedInferenceDataset(input_dataset,
                                           self.device,
                                           self.device,
                                           self.config["data"]["n_shape_components"],
                                           PoseRepresentation.from_string(self.config['conversion']['input_rotation_representation']),
                                           translation_key,
                                           shape_key,
                                           pose_key,
                                           subsampling_factor
                                           )
        data_loader = torch.utils.data.DataLoader(dataset,
                                                  batch_size=batch_size)
        if tqdm_usage == TqdmUsage.SHELL:
            loop = tqdm
        elif tqdm_usage == TqdmUsage.NOTEBOOK:
            loop = tqdm_notebook
        all_predictions = None
        n_total_elements = len(dataset)
        if errors_to_calculate is not None:
            total_metrics = torch.zeros((1 if isinstance(errors_to_calculate, str) else len(errors_to_calculate)), dtype=torch.float32)

        highest_error = torch.zeros_like(total_metrics) - 1.0
        conversion_start = perf_counter()
        for batch_idx, X in enumerate(data_loader) if tqdm_usage == TqdmUsage.OFF else enumerate(loop(data_loader, desc="Conversion")):
            bs = X.shape[0]
            bw = bs / n_total_elements
            pred, metrics = self.predict(X, False, errors_to_calculate)
            highest_error = torch.max(metrics, highest_error)
            if errors_to_calculate is not None:
                total_metrics += (metrics * bw)
            if output_file is not None:
                self._append_predictions_to_file(pred, output_file)
            if return_results:
                if all_predictions is None:
                    all_predictions = pred.cpu()
                else:
                    all_predictions = torch.vstack((all_predictions, pred.cpu()))
        conversion_time = perf_counter() - conversion_start
        print(f"Finished conversion after {conversion_time} seconds.")
        print("Highest batch error:")
        for i, metric in enumerate(errors_to_calculate) if isinstance(errors_to_calculate, list) else enumerate([errors_to_calculate]):
            print(f"Metric {metric}: {highest_error[i]}")
        if errors_to_calculate is not None:
            print("Metrics over whole dataset:")
            m = {'runtime_s': conversion_time}
            if isinstance(errors_to_calculate, str):
                m[errors_to_calculate] = total_metrics[0]
            else:
                for idx in range(len(errors_to_calculate)):
                    m[errors_to_calculate[idx]] = total_metrics[idx]
            for name, val in m.items():
                print(f"Metric {name}: {val}")
            if output_file is not None:
                metric_fname = osp.join(osp.dirname(output_file), osp.splitext(osp.basename(output_file))[0] + '_metrics.pkl')
                with open(metric_fname, 'wb') as file:
                    pickle.dump(m, file, protocol=pickle.HIGHEST_PROTOCOL)
                print(f"Conversion metrics have been written to {metric_fname}")
        else:
            m = {'runtime_s': conversion_time}
        if output_file:
            print(f"Conversion results have been written to {output_file}")
        return all_predictions, m


    def _append_predictions_to_file(self,
                                    pred: torch.Tensor,
                                    fname: str,
                                    delete_if_exists: bool = False
                                    ):
        """
            Writes the predicted data to a .npy file
        """
        if osp.splitext(fname)[1] != '.npy':
            fname += '.npy'

        with NpyAppendArray(fname, delete_if_exists=delete_if_exists) as file:
            file.append(pred.cpu().numpy())

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Parameter conversion using a"
                                     "previously trained checkpoint")
    parser.add_argument('-c', '--config', type=str, required=True,
                        help="Path to the configuration file.")
    args = parser.parse_args()
    if not osp.isfile(args.config):
        raise ValueError(f"Config file does not exist: {args.config}")
    with open(args.config, 'r') as file:
        config = yaml.safe_load(file)
    tqdm_usage = TqdmUsage.from_string(config['tqdm_usage'])
    if tqdm_usage is None:
        tqdm_usage = TqdmUsage.OFF
    converter = CombinedPredictor(config["checkpoint"],
                                  config["calculate_errors"],
                                  config["device"],
                                  config["verbosity"],
                                  config["body_model_location"],
                                  config["transfer_file_location"],
                                  config["batch_size"]
                                  )
    _, conv_metrics = converter.predict_dataset(config['dataset'],
                                                config['output_file'],
                                                False,
                                                config['translation_key'],
                                                config['shape_key'],
                                                config['pose_key'],
                                                config['batch_size'],
                                                config['subsampling_factor'],
                                                tqdm_usage,
                                                config['errors_to_calculate']
                                                )
