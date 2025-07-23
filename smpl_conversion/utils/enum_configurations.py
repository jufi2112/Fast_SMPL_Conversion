# Copyright (c) 2025 Julien Fischer, Chair of Computer Graphics and Visualization, TUD Dresden University of Technology.
# All rights reserved.
# Licensed under the MIT License. See LICENSE in the project root for license information.

import smplx
from supr.pytorch.supr import SUPR
from star.pytorch.star import STAR

from torch import nn
from enum import Enum
from typing import Union, List, Type


class TqdmUsage(Enum):
    OFF = 1
    SHELL = 2
    NOTEBOOK = 3

    @classmethod
    def from_string(cls, str_rep: str) -> Union[None, "TqdmUsage"]:
        if not isinstance(str_rep, str):
            raise ValueError("str_rep is not a string")
        if str_rep.lower() == 'off':
            return TqdmUsage.OFF
        elif str_rep.lower() == 'shell':
            return TqdmUsage.SHELL
        elif str_rep.lower() == 'notebook':
            return TqdmUsage.NOTEBOOK
        else:
            return None



class ShapeUsage(Enum):
    IGNORE = 0
    USE_GT = 1
    LEARN = 2

    @classmethod
    def from_string(cls, str_rep: str) -> Union[None, "ShapeUsage"]:
        if not isinstance(str_rep, str):
            raise ValueError("str_rep is not a string")
        if str_rep.lower() == 'ignore':
            return ShapeUsage.IGNORE
        elif str_rep.lower() == 'use_gt':
            return ShapeUsage.USE_GT
        elif str_rep.lower() == 'learn':
            return ShapeUsage.LEARN
        else:
            return None



class ExperimentType(Enum):
    COMBINED_TRAINING = 0
    COMBINED_EVALUATION = 1
    DIRECT_PARAMETER_TRANSFER_EVALUATION = 2
    FINETUNING = 3

    @classmethod
    def from_string(cls, str_rep: str) -> Union[None, "ExperimentType"]:
        if not isinstance(str_rep, str):
            raise ValueError("str_rep is not a string")
        if str_rep.lower() in ['combined_training', 'combined training', 'comb_train']:
            return ExperimentType.COMBINED_TRAINING
        elif str_rep.lower() in ['combined_evaluation', 'combined evaluation', 'comb_eval']:
            return ExperimentType.COMBINED_EVALUATION
        elif str_rep.lower() in ['direct_parameter_transfer_evaluation', 'direct parameter transfer evaluation']:
            return ExperimentType.DIRECT_PARAMETER_TRANSFER_EVALUATION
        elif str_rep.lower() in ['finetuning']:
            return ExperimentType.FINETUNING
        else:
            return None


    def to_string(self):
        if self == ExperimentType.COMBINED_TRAINING:
            return "Combined Training"
        elif self == ExperimentType.COMBINED_EVALUATION:
            return "Combined Evaluation"
        elif self == ExperimentType.DIRECT_PARAMETER_TRANSFER_EVALUATION:
            return "Direct Parameter Transfer Evaluation"
        elif self == ExperimentType.FINETUNING:
            return "Finetuning"
        else:
            return "Undefined / Not implemented"


    def __str__(self):
        return self.to_string()


    def __repr__(self):
        if self == ExperimentType.COMBINED_TRAINING:
            return "ExperimentType.COMBINED_TRAINING"
        elif self == ExperimentType.COMBINED_EVALUATION:
            return "ExperimentType.COMBINED_EVALUATION"
        elif self == ExperimentType.DIRECT_PARAMETER_TRANSFER_EVALUATION:
            return "ExperimentType.DIRECT_PARAMETER_TRANSFER_EVALUATION"
        elif self == ExperimentType.FINETUNING:
            return "ExperimentType.FINETUNING"



class PoseRepresentation(Enum):
    ROTATION_VECTOR = 0
    QUATERNION = 1
    ROTATION_MATRIX_6D = 2
    ROTATION_MATRIX_9D = 3

    @classmethod
    def from_string(cls, str_rep: str) -> Union[None, "PoseRepresentation"]:
        if str_rep is None:
            return None
        if not isinstance(str_rep, str):
            raise ValueError("str_rep is not a string")
        if str_rep.lower() == 'rot_vec' or str_rep.lower() == 'rotation_vector' or str_rep.lower() == 'rotation vector':
            return PoseRepresentation.ROTATION_VECTOR
        elif str_rep.lower() == 'quat' or str_rep.lower() == 'quaternion':
            return PoseRepresentation.QUATERNION
        elif str_rep.lower() == 'rot_mat_6d' or str_rep.lower() == 'rotation_matrix_6d' or str_rep.lower() == 'rotation matrix 6d representation':
            return PoseRepresentation.ROTATION_MATRIX_6D
        elif str_rep.lower() == 'rot_mat_9d' or str_rep.lower() == 'rotation_matrix_9d' or str_rep.lower() == 'rotation matrix 9d representation':
            return PoseRepresentation.ROTATION_MATRIX_9D
        else:
            return None


    def get_number_components(self) -> int:
        if self == PoseRepresentation.ROTATION_VECTOR:
            return 3
        elif self == PoseRepresentation.QUATERNION:
            return 4
        elif self == PoseRepresentation.ROTATION_MATRIX_6D:
            return 6
        elif self == PoseRepresentation.ROTATION_MATRIX_9D:
            return 9
        else:
            return 0


    def to_string(self):
        if self == PoseRepresentation.ROTATION_VECTOR:
            return "Rotation Vector"
        elif self == PoseRepresentation.QUATERNION:
            return "Quaternion"
        elif self == PoseRepresentation.ROTATION_MATRIX_6D:
            return "Rotation Matrix 6D Representation"
        elif self == PoseRepresentation.ROTATION_MATRIX_9D:
            return "Rotation Matrix 9D Representation"


    def __str__(self):
        return self.to_string()


    def __repr__(self):
        if self == PoseRepresentation.ROTATION_VECTOR:
            return "PoseRepresentation.ROTATION_VECTOR"
        elif self == PoseRepresentation.QUATERNION:
            return "PoseRepresentation.QUATERNION"
        elif self == PoseRepresentation.ROTATION_MATRIX_6D:
            return "PoseRepresentation.ROTATION_MATRIX_6D"
        elif self == PoseRepresentation.ROTATION_MATRIX_9D:
            return "PoseRepresentation.ROTATION_MATRIX_9D"


class ConversionNetworkArchitecture(Enum):
    COMBINED_SINGLE = 0
    COMBINED_PER_PARAMETER = 1
    COMBINED_SKIP = 2
    COMBINED_SKIP_LINEAR = 3
    COMBINED_SKIP_WITHOUT_UPSAMPLING = 4
    SEPARATED_SIMPLE = 5
    SEPARATED_SKIP = 6
    SEPARATED_ATTENTION = 7
    SEPARATED_ATTENTION_SKIP = 8
    SEPARATED_PER_JOINT = 9
    SEPARATED_PER_JOINT_COMBINED = 10


    @classmethod
    def from_string(cls, str_rep: str) -> Union[None, "ConversionNetworkArchitecture"]:
        if str_rep is None:
            return None
        if not isinstance(str_rep, str):
            raise ValueError("str_rep is not a string")
        # To support checkpoints saved before the changes, also treat 'single' as combined_single
        if str_rep.lower() in ['combined_single', 'combined single', 'single']:
            return ConversionNetworkArchitecture.COMBINED_SINGLE
        elif str_rep.lower() in ['combined_per_parameter', 'combined per parameter']:
            return ConversionNetworkArchitecture.COMBINED_PER_PARAMETER
        elif str_rep.lower() in ['combined_skip', 'combined skip']:
            return ConversionNetworkArchitecture.COMBINED_SKIP
        elif str_rep.lower() in ['combined_skip_linear', 'combined skip linear']:
            return ConversionNetworkArchitecture.COMBINED_SKIP_LINEAR
        elif str_rep.lower() in ['combined_skip_without_upsampling', 'combined skip without upsampling']:
            return ConversionNetworkArchitecture.COMBINED_SKIP_WITHOUT_UPSAMPLING
        elif str_rep.lower() in ['separated_simple', 'separated simple']:
            return ConversionNetworkArchitecture.SEPARATED_SIMPLE
        elif str_rep.lower() in ['separated_skip', 'separated skip']:
            return ConversionNetworkArchitecture.SEPARATED_SKIP
        elif str_rep.lower() in ['separated_attention', 'separated attention']:
            return ConversionNetworkArchitecture.SEPARATED_ATTENTION
        elif str_rep.lower() in ['separated_attention_skip', 'separated attention skip']:
            return ConversionNetworkArchitecture.SEPARATED_ATTENTION_SKIP
        elif str_rep.lower() in ['separated_per_joint', 'separated per joint']:
            return ConversionNetworkArchitecture.SEPARATED_PER_JOINT
        elif str_rep.lower() in ['separated_per_joint_combined', 'separated per joint combined']:
            return ConversionNetworkArchitecture.SEPARATED_PER_JOINT_COMBINED
        else:
            return None


    def to_string(self):
        if self == ConversionNetworkArchitecture.COMBINED_SINGLE:
            return "Combined Single"
        elif self == ConversionNetworkArchitecture.COMBINED_PER_PARAMETER:
            return "Combined Per Parameter"
        elif self == ConversionNetworkArchitecture.COMBINED_SKIP:
            return "Combined Skip"
        elif self == ConversionNetworkArchitecture.COMBINED_SKIP_LINEAR:
            return "Combined Skip Linear"
        elif self == ConversionNetworkArchitecture.COMBINED_SKIP_WITHOUT_UPSAMPLING:
            return "Combined Skip Without Upsampling"
        elif self == ConversionNetworkArchitecture.SEPARATED_SIMPLE:
            return "Separated Simple"
        elif self == ConversionNetworkArchitecture.SEPARATED_SKIP:
            return "Separated Skip"
        elif self == ConversionNetworkArchitecture.SEPARATED_ATTENTION:
            return "Separated Attention"
        elif self == ConversionNetworkArchitecture.SEPARATED_ATTENTION_SKIP:
            return "Separated Attention Skip"
        elif self == ConversionNetworkArchitecture.SEPARATED_PER_JOINT:
            return "Separated Per Joint"
        elif self == ConversionNetworkArchitecture.SEPARATED_PER_JOINT_COMBINED:
            return "Separated Per Joint Combined"


    def is_combined_architecture(self) -> bool:
        """
            Whether the architecture is a combined one.
        """
        if self in [ConversionNetworkArchitecture.COMBINED_SINGLE,
                    ConversionNetworkArchitecture.COMBINED_PER_PARAMETER,
                    ConversionNetworkArchitecture.COMBINED_SKIP,
                    ConversionNetworkArchitecture.COMBINED_SKIP_LINEAR,
                    ConversionNetworkArchitecture.COMBINED_SKIP_WITHOUT_UPSAMPLING]:
            return True
        else:
            return False


    def is_separated_architecture(self) -> bool:
        """
            Whether the architecture is a separated one.

            Note
            ----
                Separated Per Joint should not be included in these
        """
        if self in [ConversionNetworkArchitecture.SEPARATED_SIMPLE,
                    ConversionNetworkArchitecture.SEPARATED_SKIP,
                    ConversionNetworkArchitecture.SEPARATED_ATTENTION,
                    ConversionNetworkArchitecture.SEPARATED_ATTENTION_SKIP]:
            return True
        else:
            return False


    def __str__(self):
        return self.to_string()


    def __repr__(self):
        if self == ConversionNetworkArchitecture.COMBINED_SINGLE:
            return "ConversionNetworkArchitecture.COMBINED_SINGLE"
        elif self == ConversionNetworkArchitecture.COMBINED_PER_PARAMETER:
            return "ConversionNetworkArchitecture.COMBINED_PER_PARAMETER"
        elif self == ConversionNetworkArchitecture.COMBINED_SKIP:
            return "ConversionNetworkArchitecture.COMBINED_SKIP"
        elif self == ConversionNetworkArchitecture.COMBINED_SKIP_LINEAR:
            return "ConversionNetworkArchitecture.COMBINED_SKIP_LINEAR"
        elif self == ConversionNetworkArchitecture.COMBINED_SKIP_WITHOUT_UPSAMPLING:
            return "ConversionNetworkArchitecture.COMBINED_SKIP_WITHOUT_UPSAMPLING"
        elif self == ConversionNetworkArchitecture.SEPARATED_SIMPLE:
            return "ConversionNetworkArchitecture.SEPARATED_SIMPLE"
        elif self == ConversionNetworkArchitecture.SEPARATED_SKIP:
            return "ConversionNetworkArchitecture.SEPARATED_SKIP"
        elif self == ConversionNetworkArchitecture.SEPARATED_ATTENTION:
            return "ConversionNetworkArchitecture.SEPARATED_ATTENTION"
        elif self == ConversionNetworkArchitecture.SEPARATED_ATTENTION_SKIP:
            return "ConversionNetworkArchitecture.SEPARATED_ATTENTION_SKIP"
        elif self == ConversionNetworkArchitecture.SEPARATED_PER_JOINT:
            return "ConversionNetworkArchitecture.SEPARATED_PER_JOINT"
        elif self == ConversionNetworkArchitecture.SEPARATED_PER_JOINT_COMBINED:
            return "ConversionNetworkArchitecture.SEPARATED_PER_JOINT_COMBINED"



class PositionalEncodingMode(Enum):
    FIXED = 1
    LEARNABLE = 2


    @classmethod
    def from_string(cls, str_rep: str) -> Union[None, "PositionalEncodingMode"]:
        if str_rep is None:
            return None
        if not isinstance(str_rep, str):
            raise ValueError("str_rep is not a string")
        if str_rep.lower() == 'fixed':
            return PositionalEncodingMode.FIXED
        elif str_rep.lower() == 'learnable':
            return PositionalEncodingMode.LEARNABLE
        else:
            return None


    def to_string(self):
        if self == PositionalEncodingMode.FIXED:
            return "Fixed"
        elif self == PositionalEncodingMode.LEARNABLE:
            return "Learnable"


    def __str__(self):
        return self.to_string()


    def __repr__(self):
        if self == PositionalEncodingMode.FIXED:
            return "PositionalEncodingMode.FIXED"
        elif self == PositionalEncodingMode.LEARNABLE:
            return "PositionalEncodingMode.LEARNABLE"



class NetworkNormalizationMode(Enum):
    LAYER_NORM=1
    BATCH_NORM=2
    NONE=3


    @classmethod
    def from_string(cls, str_rep: str) -> Union["NetworkNormalizationMode", None]:
        if str_rep is None:
            return None
        if not isinstance(str_rep, str):
            raise ValueError("str_rep is not a string!")
        if str_rep.lower() in ['layernorm', 'layer_norm']:
            return NetworkNormalizationMode.LAYER_NORM
        elif str_rep.lower() in ['batchnorm', 'batch_norm']:
            return NetworkNormalizationMode.BATCH_NORM
        elif str_rep.lower() == "none":
            return NetworkNormalizationMode.NONE
        else:
            return None


    def to_string(self):
        if self == NetworkNormalizationMode.LAYER_NORM:
            return "Layernorm"
        elif self == NetworkNormalizationMode.BATCH_NORM:
            return "Batchnorm"
        elif self == NetworkNormalizationMode.NONE:
            return "None"


    def __str__(self):
        return self.to_string()


    def __repr__(self):
        if self == NetworkNormalizationMode.LAYER_NORM:
            return "NetworkNormalizationMode.LAYER_NORM"
        elif self == NetworkNormalizationMode.BATCH_NORM:
            return "NetworkNormalizationMode.BATCH_NORM"
        elif self == NetworkNormalizationMode.NONE:
            return "NetworkNormalizationMode.NONE"


    def get_pytorch_layer_class(self):
        if self == NetworkNormalizationMode.LAYER_NORM:
            return nn.LayerNorm
        elif self == NetworkNormalizationMode.BATCH_NORM:
            return nn.BatchNorm1d
        elif self == NetworkNormalizationMode.NONE:
            return nn.Identity



class NetworkActivationFunction(Enum):
    LEAKY_RELU=1
    RELU=2


    @classmethod
    def from_string(cls, str_rep: str) -> Union["NetworkActivationFunction", None]:
        if str_rep is None:
            return None
        if not isinstance(str_rep, str):
            raise ValueError("str_rep is not a string!")
        if str_rep.lower() in ['leaky_relu', 'leaky relu', 'leakyrelu']:
            return NetworkActivationFunction.LEAKY_RELU
        elif str_rep.lower() in ['relu']:
            return NetworkActivationFunction.RELU
        else:
            return None


    def to_string(self):
        if self == NetworkActivationFunction.LEAKY_RELU:
            return "Leaky ReLU"
        if self == NetworkActivationFunction.RELU:
            return "ReLU"


    def __str__(self):
        return self.to_string()


    def __repr__(self):
        if self == NetworkActivationFunction.LEAKY_RELU:
            return "NetworkActivationFunction.LEAKY_RELU"
        elif self == NetworkActivationFunction.RELU:
            return "NetworkActivationFunction.RELU"


    def get_pytorch_layer_class(self):
        if self == NetworkActivationFunction.LEAKY_RELU:
            return nn.LeakyReLU
        elif self == NetworkActivationFunction.RELU:
            return nn.ReLU



class MetricReturnType(Enum):
    ALL=1
    MEAN=2
    SUM=3


    @classmethod
    def from_string(cls, str_rep: str) -> Union["MetricReturnType", None]:
        if str_rep is None:
            return None
        if not isinstance(str_rep, str):
            raise ValueError("str_rep is not a string!")
        if str_rep.lower() == 'all':
            return MetricReturnType.ALL
        elif str_rep.lower() == 'mean':
            return MetricReturnType.MEAN
        elif str_rep.lower() == 'sum':
            return MetricReturnType.SUM
        else:
            return None


    def to_string(self):
        if self == MetricReturnType.ALL:
            return "All"
        elif self == MetricReturnType.MEAN:
            return "Mean"
        elif self == MetricReturnType.SUM:
            return "Sum"


    def __str__(self):
        return self.to_string()


    def __repr__(self):
        if self == MetricReturnType.ALL:
            return "MetricReturnType.ALL"
        elif self == MetricReturnType.MEAN:
            return "MetricReturnType.MEAN"
        elif self == MetricReturnType.SUM:
            return "MetricReturnType.SUM"



class BodyModelType(Enum):
    SMPL=1
    SMPLH=2
    SMPLX=3
    STAR=4
    SUPR=5
    SKEL=6


    @classmethod
    def from_string(cls, str_rep: str) -> Union["BodyModelType", None]:
        if str_rep is None:
            return None
        if not isinstance(str_rep, str):
            raise ValueError("str_rep is not a string!")
        if str_rep.lower() in ['smpl']:
            return BodyModelType.SMPL
        elif str_rep.lower() in ['smplh', 'smpl+h']:
            return BodyModelType.SMPLH
        elif str_rep.lower() in ['smplx', 'smpl-x']:
            return BodyModelType.SMPLX
        elif str_rep.lower() in ['star']:
            return BodyModelType.STAR
        elif str_rep.lower() in ['supr']:
            return BodyModelType.SUPR
        elif str_rep.lower() in ['skel']:
            return BodyModelType.SKEL
        else:
            return None


    @classmethod
    def get_supported_body_model_types(cls) -> List["BodyModelType"]:
        return [BodyModelType.SMPL, BodyModelType.SMPLH, BodyModelType.SMPLX, BodyModelType.SUPR, BodyModelType.STAR]


    @classmethod
    def from_body_model_instance(cls, body_model_instance: nn.Module) -> Union["BodyModelType", None]:
        if type(body_model_instance) == smplx.SMPL or type(body_model_instance) == smplx.SMPLLayer:
            return BodyModelType.SMPL
        elif type(body_model_instance) == smplx.SMPLH or type(body_model_instance) == smplx.SMPLHLayer:
            return BodyModelType.SMPLH
        elif type(body_model_instance) == smplx.SMPLX or type(body_model_instance) == smplx.SMPLXLayer:
            return BodyModelType.SMPLX
        elif type(body_model_instance) == SUPR:
            return BodyModelType.SUPR
        elif type(body_model_instance) == STAR:
            return BodyModelType.STAR
        else:
            return None


    def get_model_class(self) -> Type:
        if self == BodyModelType.SMPL:
            return smplx.SMPL
        elif self == BodyModelType.SMPLH:
            return smplx.SMPLH
        elif self == BodyModelType.SMPLX:
            return smplx.SMPLX
        elif self == BodyModelType.SUPR:
            return SUPR
        elif self == BodyModelType.STAR:
            return STAR
        elif self == BodyModelType.SKEL:
            raise NotImplementedError("SKEL is currently not implemented!")


    def to_string(self) -> str:
        if self == BodyModelType.SMPL:
            return "SMPL"
        elif self == BodyModelType.SMPLH:
            return "SMPL+H"
        elif self == BodyModelType.SMPLX:
            return "SMPL-X"
        elif self == BodyModelType.STAR:
            return "STAR"
        elif self == BodyModelType.SUPR:
            return "SUPR"
        elif self == BodyModelType.SKEL:
            return "SKEL"


    def to_internal_string(self) -> str:
        if self == BodyModelType.SMPL:
            return "smpl"
        elif self == BodyModelType.SMPLH:
            return "smplh"
        elif self == BodyModelType.SMPLX:
            return "smplx"
        elif self == BodyModelType.STAR:
            return "star"
        elif self == BodyModelType.SUPR:
            return "supr"
        elif self == BodyModelType.SKEL:
            return "skel"


    def __str__(self) -> str:
        return self.to_string()


    def __repr__(self) -> str:
        if self == BodyModelType.SMPL:
            return "BodyModelType.SMPL"
        elif self == BodyModelType.SMPLH:
            return "BodyModelType.SMPLH"
        elif self == BodyModelType.SMPLX:
            return "BodyModelType.SMPLX"
        elif self == BodyModelType.STAR:
            return "BodyModelType.STAR"
        elif self == BodyModelType.SUPR:
            return "BodyModelType.SUPR"
        elif self == BodyModelType.SKEL:
            return "BodyModelType.SKEL"



class BodyPart(Enum):
    HIPS=1
    SPINE=2
    LEFT_UP_LEG=3
    RIGHT_UP_LEG=4
    SPINE_1=5
    LEFT_LEG=6
    RIGHT_LEG=7
    SPINE_2=8
    LEFT_FOOT=9
    LEFT_TOE_BASE=10
    RIGHT_FOOT=11
    RIGHT_TOE_BASE=12
    NECK=13
    HEAD=14
    LEFT_SHOULDER=15
    RIGHT_SHOULDER=16
    LEFT_ARM=17
    RIGHT_ARM=18
    LEFT_FORE_ARM=19
    RIGHT_FORE_ARM=20
    LEFT_HAND=21
    RIGHT_HAND=22
    LEFT_HAND_INDEX_1=23
    RIGHT_HAND_INDEX_1=24
    LEFT_EYE=25
    RIGHT_EYE=26
    EYEBALLS=27


    @classmethod
    def get_possible_names(cls, base_name: str) -> List[str]:
        possible_names = [base_name]
        # replace underscores by whitespace
        possible_names.append(base_name.replace("_", " "))
        # remove underscores
        possible_names.append(base_name.replace("_", ""))
        return possible_names


    @classmethod
    def from_string(cls, str_rep: str) -> Union["BodyPart", None]:
        if str_rep is None:
            return None
        if not isinstance(str_rep, str):
            raise ValueError("str_rep is not a string!")
        if str_rep.lower() in cls.get_possible_names('hips'):
            return BodyPart.HIPS
        elif str_rep.lower() in cls.get_possible_names('spine'):
            return BodyPart.SPINE
        elif str_rep.lower() in cls.get_possible_names('left_up_leg'):
            return BodyPart.LEFT_UP_LEG
        elif str_rep.lower() in cls.get_possible_names('right_up_leg'):
            return BodyPart.RIGHT_UP_LEG
        elif str_rep.lower() in cls.get_possible_names('spine_1'):
            return BodyPart.SPINE_1
        elif str_rep.lower() in cls.get_possible_names('left_leg'):
            return BodyPart.LEFT_LEG
        elif str_rep.lower() in cls.get_possible_names('right_leg'):
            return BodyPart.RIGHT_LEG
        elif str_rep.lower() in cls.get_possible_names('spine_2'):
            return BodyPart.SPINE_2
        elif str_rep.lower() in cls.get_possible_names('left_foot'):
            return BodyPart.LEFT_FOOT
        elif str_rep.lower() in cls.get_possible_names('left_toe_base'):
            return BodyPart.LEFT_TOE_BASE
        elif str_rep.lower() in cls.get_possible_names('right_foot'):
            return BodyPart.RIGHT_FOOT
        elif str_rep.lower() in cls.get_possible_names('right_toe_base'):
            return BodyPart.RIGHT_TOE_BASE
        elif str_rep.lower() in cls.get_possible_names('neck'):
            return BodyPart.NECK
        elif str_rep.lower() in cls.get_possible_names('head'):
            return BodyPart.HEAD
        elif str_rep.lower() in cls.get_possible_names('left_shoulder'):
            return BodyPart.LEFT_SHOULDER
        elif str_rep.lower() in cls.get_possible_names('right_shoulder'):
            return BodyPart.RIGHT_SHOULDER
        elif str_rep.lower() in cls.get_possible_names('left_arm'):
            return BodyPart.LEFT_ARM
        elif str_rep.lower() in cls.get_possible_names('right_arm'):
            return BodyPart.RIGHT_ARM
        elif str_rep.lower() in cls.get_possible_names('left_fore_arm'):
            return BodyPart.LEFT_FORE_ARM
        elif str_rep.lower() in cls.get_possible_names('right_fore_arm'):
            return BodyPart.RIGHT_FORE_ARM
        elif str_rep.lower() in cls.get_possible_names('left_hand'):
            return BodyPart.LEFT_HAND
        elif str_rep.lower() in cls.get_possible_names('right_hand'):
            return BodyPart.RIGHT_HAND
        elif str_rep.lower() in cls.get_possible_names('left_hand_index_1'):
            return BodyPart.LEFT_HAND_INDEX_1
        elif str_rep.lower() in cls.get_possible_names('right_hand_index_1'):
            return BodyPart.RIGHT_HAND_INDEX_1
        elif str_rep.lower() in cls.get_possible_names('left_eye'):
            return BodyPart.LEFT_EYE
        elif str_rep.lower() in cls.get_possible_names('right_eye'):
            return BodyPart.RIGHT_EYE
        elif str_rep.lower() in cls.get_possible_names('eyeballs'):
            return BodyPart.EYEBALLS
        else:
            return None


    @classmethod
    def get_smpl_body_parts(cls) -> List["BodyPart"]:
        return [
            BodyPart.HIPS,
            BodyPart.SPINE,
            BodyPart.LEFT_UP_LEG,
            BodyPart.RIGHT_UP_LEG,
            BodyPart.SPINE_1,
            BodyPart.LEFT_LEG,
            BodyPart.RIGHT_LEG,
            BodyPart.SPINE_2,
            BodyPart.LEFT_FOOT,
            BodyPart.LEFT_TOE_BASE,
            BodyPart.RIGHT_FOOT,
            BodyPart.RIGHT_TOE_BASE,
            BodyPart.NECK,
            BodyPart.HEAD,
            BodyPart.LEFT_SHOULDER,
            BodyPart.RIGHT_SHOULDER,
            BodyPart.LEFT_ARM,
            BodyPart.RIGHT_ARM,
            BodyPart.LEFT_FORE_ARM,
            BodyPart.RIGHT_FORE_ARM,
            BodyPart.LEFT_HAND,
            BodyPart.RIGHT_HAND,
            BodyPart.LEFT_HAND_INDEX_1,
            BodyPart.RIGHT_HAND_INDEX_1
        ]


    @classmethod
    def get_smplx_body_parts(cls) -> List["BodyPart"]:
        return [
            BodyPart.HIPS,
            BodyPart.SPINE,
            BodyPart.LEFT_UP_LEG,
            BodyPart.RIGHT_UP_LEG,
            BodyPart.SPINE_1,
            BodyPart.LEFT_LEG,
            BodyPart.RIGHT_LEG,
            BodyPart.SPINE_2,
            BodyPart.LEFT_FOOT,
            BodyPart.LEFT_TOE_BASE,
            BodyPart.RIGHT_FOOT,
            BodyPart.RIGHT_TOE_BASE,
            BodyPart.NECK,
            BodyPart.HEAD,
            BodyPart.LEFT_SHOULDER,
            BodyPart.RIGHT_SHOULDER,
            BodyPart.LEFT_ARM,
            BodyPart.RIGHT_ARM,
            BodyPart.LEFT_FORE_ARM,
            BodyPart.RIGHT_FORE_ARM,
            BodyPart.LEFT_HAND,
            BodyPart.RIGHT_HAND,
            BodyPart.LEFT_HAND_INDEX_1,
            BodyPart.RIGHT_HAND_INDEX_1,
            BodyPart.LEFT_EYE,
            BodyPart.RIGHT_EYE,
            BodyPart.EYEBALLS
        ]


    @classmethod
    def get_body_parts_for_body_model(cls, model_type: BodyModelType):
        if model_type in [BodyModelType.SMPL, BodyModelType.SMPLH,
                          BodyModelType.STAR, BodyModelType.SKEL]:
            return cls.get_smpl_body_parts()
        elif model_type in [BodyModelType.SMPLX, BodyModelType.SUPR]:
            return cls.get_smplx_body_parts()
        else:
            raise ValueError(f"Body model type {model_type} is not supported!")


    def to_string(self) -> str:
        if self == BodyPart.HIPS:
            return "Hips"
        elif self == BodyPart.SPINE:
            return "Spine"
        elif self == BodyPart.LEFT_UP_LEG:
            return "Left Up Leg"
        elif self == BodyPart.RIGHT_UP_LEG:
            return "Right Up Leg"
        elif self == BodyPart.SPINE_1:
            return "Spine 1"
        elif self == BodyPart.LEFT_LEG:
            return "Left Leg"
        elif self == BodyPart.RIGHT_LEG:
            return "Right Leg"
        elif self == BodyPart.SPINE_2:
            return "Spine 2"
        elif self == BodyPart.LEFT_FOOT:
            return "Left Foot"
        elif self == BodyPart.LEFT_TOE_BASE:
            return "Left Toe Base"
        elif self == BodyPart.RIGHT_FOOT:
            return "Right Foot"
        elif self == BodyPart.RIGHT_TOE_BASE:
            return "Right Toe Base"
        elif self == BodyPart.NECK:
            return "Neck"
        elif self == BodyPart.HEAD:
            return "Head"
        elif self == BodyPart.LEFT_SHOULDER:
            return "Left Shoulder"
        elif self == BodyPart.RIGHT_SHOULDER:
            return "Right Shoulder"
        elif self == BodyPart.LEFT_ARM:
            return "Left Arm"
        elif self == BodyPart.RIGHT_ARM:
            return "Right Arm"
        elif self == BodyPart.LEFT_FORE_ARM:
            return "Left Fore Arm"
        elif self == BodyPart.RIGHT_FORE_ARM:
            return "Right Fore Arm"
        elif self == BodyPart.LEFT_HAND:
            return "Left Hand"
        elif self == BodyPart.RIGHT_HAND:
            return "Right Hand"
        elif self == BodyPart.LEFT_HAND_INDEX_1:
            return "Left Hand Index 1"
        elif self == BodyPart.RIGHT_HAND_INDEX_1:
            return "Right Hand Index 1"
        elif self == BodyPart.LEFT_EYE:
            return "Left Eye"
        elif self == BodyPart.RIGHT_EYE:
            return "Right Eye"
        elif self == BodyPart.EYEBALLS:
            return "Eyeballs"


    def __str__(self) -> str:
        return self.to_string()


    def __repr__(self) -> str:
        if self == BodyPart.HIPS:
            return "BodyPart.HIPS"
        elif self == BodyPart.SPINE:
            return "BodyPart.SPINE"
        elif self == BodyPart.LEFT_UP_LEG:
            return "BodyPart.LEFT_UP_LEG"
        elif self == BodyPart.RIGHT_UP_LEG:
            return "BodyPart.RIGHT_UP_LEG"
        elif self == BodyPart.SPINE_1:
            return "BodyPart.SPINE_1"
        elif self == BodyPart.LEFT_LEG:
            return "BodyPart.LEFT_LEG"
        elif self == BodyPart.RIGHT_LEG:
            return "BodyPart.RIGHT_LEG"
        elif self == BodyPart.SPINE_2:
            return "BodyPart.SPINE_2"
        elif self == BodyPart.LEFT_FOOT:
            return "BodyPart.LEFT_FOOT"
        elif self == BodyPart.LEFT_TOE_BASE:
            return "BodyPart.LEFT_TOE_BASE"
        elif self == BodyPart.RIGHT_FOOT:
            return "BodyPart.RIGHT_FOOT"
        elif self == BodyPart.RIGHT_TOE_BASE:
            return "BodyPart.RIGHT_TOE_BASE"
        elif self == BodyPart.NECK:
            return "BodyPart.NECK"
        elif self == BodyPart.HEAD:
            return "BodyPart.HEAD"
        elif self == BodyPart.LEFT_SHOULDER:
            return "BodyPart.LEFT_SHOULDER"
        elif self == BodyPart.RIGHT_SHOULDER:
            return "BodyPart.RIGHT_SHOULDER"
        elif self == BodyPart.LEFT_ARM:
            return "BodyPart.LEFT_ARM"
        elif self == BodyPart.RIGHT_ARM:
            return "BodyPart.RIGHT_ARM"
        elif self == BodyPart.LEFT_FORE_ARM:
            return "BodyPart.LEFT_FORE_ARM"
        elif self == BodyPart.RIGHT_FORE_ARM:
            return "BodyPart.RIGHT_FORE_ARM"
        elif self == BodyPart.LEFT_HAND:
            return "BodyPart.LEFT_HAND"
        elif self == BodyPart.RIGHT_HAND:
            return "BodyPart.RIGHT_HAND"
        elif self == BodyPart.LEFT_HAND_INDEX_1:
            return "BodyPart.LEFT_HAND_INDEX_1"
        elif self == BodyPart.RIGHT_HAND_INDEX_1:
            return "BodyPart.RIGHT_HAND_INDEX_1"
        elif self == BodyPart.LEFT_EYE:
            return "BodyPart.LEFT_EYE"
        elif self == BodyPart.RIGHT_EYE:
            return "BodyPart.RIGHT_EYE"
        elif self == BodyPart.EYEBALLS:
            return "BodyPart.EYEBALLS"
