# Copyright (c) 2025 Julien Fischer, Chair of Computer Graphics and Visualization, TUD Dresden University of Technology.
# All rights reserved.
# Licensed under the MIT License. See LICENSE in the project root for license information.

from smplx import SMPL, SMPLH, SMPLX
from supr.pytorch.supr import SUPR
from star.pytorch.star import STAR
import numpy as np
from typing import Union, List
from smpl_conversion.utils.enum_configurations import BodyModelType, PoseRepresentation


# closed intervals
MODEL_JOINT_INDICES = {
    BodyModelType.SMPL: {
        'body': (0, BodyModelType.SMPL.get_model_class().NUM_BODY_JOINTS),
        'face': (-1, -1),
        'hands': (-1, -1),
        'lhand': (-1, -1),
        'rhand': (-1, -1),
        'feet': (-1, -1),
        'lfeet': (-1, -1),
        'rfeet': (-1, -1)
    },
    BodyModelType.SMPLH: {
        'body': (0, BodyModelType.SMPLH.get_model_class().NUM_BODY_JOINTS),
        'face': (-1, -1),
        'hands': (
            BodyModelType.SMPLH.get_model_class().NUM_BODY_JOINTS + 1,
            BodyModelType.SMPLH.get_model_class().NUM_JOINTS
        ),
        'lhand': (
            BodyModelType.SMPLH.get_model_class().NUM_BODY_JOINTS + 1,
            BodyModelType.SMPLH.get_model_class().NUM_BODY_JOINTS + BodyModelType.SMPLH.get_model_class().NUM_HAND_JOINTS
        ),
        'rhand': (
            BodyModelType.SMPLH.get_model_class().NUM_BODY_JOINTS + BodyModelType.SMPLH.get_model_class().NUM_HAND_JOINTS + 1,
            BodyModelType.SMPLH.get_model_class().NUM_JOINTS
        ),
        'feet': (-1, -1),
        'lfeet': (-1, -1),
        'rfeet': (-1, -1)
    },
    BodyModelType.SMPLX: {
        'body': (0, BodyModelType.SMPLX.get_model_class().NUM_BODY_JOINTS),
        'face': (
            BodyModelType.SMPLX.get_model_class().NUM_BODY_JOINTS + 1,
            BodyModelType.SMPLX.get_model_class().NUM_BODY_JOINTS + BodyModelType.SMPLX.get_model_class().NUM_FACE_JOINTS
        ),
        'hands': (
            BodyModelType.SMPLX.get_model_class().NUM_BODY_JOINTS + BodyModelType.SMPLX.get_model_class().NUM_FACE_JOINTS + 1,
            BodyModelType.SMPLX.get_model_class().NUM_JOINTS
        ),
        'lhand': (
            BodyModelType.SMPLX.get_model_class().NUM_BODY_JOINTS + BodyModelType.SMPLX.get_model_class().NUM_FACE_JOINTS + 1,
            BodyModelType.SMPLX.get_model_class().NUM_BODY_JOINTS + BodyModelType.SMPLX.get_model_class().NUM_FACE_JOINTS + BodyModelType.SMPLX.get_model_class().NUM_HAND_JOINTS
        ),
        'rhand': (
            BodyModelType.SMPLX.get_model_class().NUM_BODY_JOINTS + BodyModelType.SMPLX.get_model_class().NUM_FACE_JOINTS + BodyModelType.SMPLX.get_model_class().NUM_HAND_JOINTS + 1,
            BodyModelType.SMPLX.get_model_class().NUM_JOINTS
        ),
        'feet': (-1, -1),
        'lfeet': (-1, -1),
        'rfeet': (-1, -1)
    },
    BodyModelType.SUPR: {
        'body': (0, 21),
        'face': (22, 24),
        'hands': (25, 54),
        'lhand': (25, 39),
        'rhand': (40, 54),
        'feet': (55, 74),
        'lfeet': (55, 64),
        'rfeet': (65, 74)
    },
    BodyModelType.STAR: {
        'body': (0, 23),
        'face': (-1, -1),
        'hands': (-1, -1),
        'lhand': (-1, -1),
        'rhand': (-1, -1),
        'feet': (-1, -1),
        'lfeet': (-1, -1),
        'rfeet': (-1, -1)
    }
}


# returns same results as MODEL_CONVERSION below, but takes into account pose representation
def direct_transfer_indices(origin: BodyModelType,
                            target: BodyModelType,
                            pose_rep: PoseRepresentation
                            ) -> Union[np.ndarray, List[int]]:
    """
        Returns an array that can be used to index all elements from origin model type
        that should be carried over to target model type.

        Params
        ------
            origin (BodyModelType):
                Model type from which data should be transferred
            target (BodyModelType):
                Model type to which data should be transferred
            pose_rep (PoseRepresentation):
                Representation in which data are available

        Returns
        -------
            np.ndarray or list of int:
                Indices that select all data in origin that can be directly
                transferred to target model type.

        Raises
        ------
            NotImplementedError:
                If no indexing from origin to target is available
    """
    pose_elems = pose_rep.get_number_components()

    if origin == BodyModelType.SUPR:
        if target in [BodyModelType.SMPL, BodyModelType.STAR]:
            return np.arange(0, 22 * pose_elems)
        elif target == BodyModelType.SMPLH:
            return list(np.arange(0, 22 * pose_elems)) + list(np.arange(25 * pose_elems, 55 * pose_elems))
        elif target == BodyModelType.SMPLX:
            return np.arange(0, 55 * pose_elems)

    elif origin == BodyModelType.SMPLX:
        if target in [BodyModelType.SMPL, BodyModelType.STAR]:
            return np.arange(0, 22 * pose_elems)
        elif target == BodyModelType.SMPLH:
            return list(np.arange(0, 22 * pose_elems)) + list(np.arange(25 * pose_elems, 55 * pose_elems))
        elif target == BodyModelType.SUPR:
            return np.arange(0, 55 * pose_elems)

    elif origin == BodyModelType.SMPLH:
        if target in [BodyModelType.SMPL, BodyModelType.STAR]:
            return np.arange(0, 22 * pose_elems)
        elif target == BodyModelType.SMPLX:
            return np.arange(0, 52 * pose_elems)
        elif target == BodyModelType.SUPR:
            return np.arange(0, 52 * pose_elems)

    elif origin == BodyModelType.SMPL:
        if target == BodyModelType.STAR:
            return np.arange(0, 24 * pose_elems)
        elif target in [BodyModelType.SMPLH, BodyModelType.SMPLX, BodyModelType.SUPR]:
            return np.arange(0, 22 * pose_elems)

    elif origin == BodyModelType.STAR:
        if target == BodyModelType.SMPL:
            return np.arange(0, 24 * pose_elems)
        elif target in [BodyModelType.SMPLH, BodyModelType.SMPLX, BodyModelType.SUPR]:
            return np.arange(0, 22 * pose_elems)

    raise NotImplementedError(f"Could not find transfer indices from model type {origin} to model type {target}")



# from (outer) to (inner)
# MODEL_CONVERSION = {
#     BodyModelType.SUPR: {
#         BodyModelType.SMPL: np.arange(0, 22*3),
#         BodyModelType.SMPLH: list(np.arange(0, 22*3)) + list(np.arange(25*3, 55*3)),
#         BodyModelType.SMPLX: np.arange(0, 55*3),
#         BodyModelType.STAR: np.arange(0, 22*3)
#     },
#     BodyModelType.SMPLX: {
#         BodyModelType.SMPL: np.arange(0, 22*3),
#         BodyModelType.SMPLH: list(np.arange(0, 22*3)) + list(np.arange(25*3, 55*3)),
#         BodyModelType.SUPR: np.arange(0, 55*3),
#         BodyModelType.STAR: np.arange(0, 22*3)
#     },
#     BodyModelType.SMPLH: {
#         BodyModelType.SMPL: np.arange(0, 22*3),
#         BodyModelType.SMPLX: np.arange(0, 52*3),
#         BodyModelType.SUPR: np.arange(0, 52*3),
#         BodyModelType.STAR: np.arange(0, 22*3)
#     },
#     BodyModelType.SMPL: {
#         BodyModelType.SMPLH: np.arange(0, 22*3),
#         BodyModelType.SMPLX: np.arange(0, 22*3),
#         BodyModelType.SUPR: np.arange(0, 22*3),
#         BodyModelType.STAR: np.arange(0, 24*3)
#     },
#     BodyModelType.STAR: {
#         BodyModelType.SMPL: np.arange(0, 24*3),
#         BodyModelType.SMPLH: np.arange(0, 22*3),
#         BodyModelType.SMPLX: np.arange(0, 22*3),
#         BodyModelType.SUPR: np.arange(0, 22*3),
#     }
# }

# Joint names: https://github.com/vchoutas/smplx/blob/main/smplx/joint_names.py
PARENT_JOINT_MAPPINGS = {
    # e.g. https://www.researchgate.net/figure/Layout-of-23-joints-in-the-SMPL-models_fig2_351179264
    BodyModelType.SMPL: {
        0: None,    # Pelvis
        # Body joints
        1: 0,       # Left Hip
        2: 0,       # Right Hip
        3: 0,       # Spine 1
        4: 1,       # Left Knee
        5: 2,       # Right Knee
        6: 3,       # Spine 2
        7: 4,       # Left Ankle
        8: 5,       # Right Ankle,
        9: 6,       # Spine 3
        10: 7,      # Left Foot
        11: 8,      # Right Foot
        12: 9,      # Neck
        13: 9,      # Left Collar
        14: 9,      # Right Collar
        15: 12,     # Head
        16: 13,     # Left Shoulder
        17: 14,     # Right Shoulder
        18: 16,     # Left Ellbow
        19: 17,     # Right Ellbow
        20: 18,     # Left Wrist
        21: 19,     # Right Wrist
        22: 20,     # Left Hand
        23: 21      # Right Hand
    },
    BodyModelType.SMPLH: {
        0: None,    # Pelvis
        # Body joints
        1: 0,       # Left Hip
        2: 0,       # Right Hip
        3: 0,       # Spine 1
        4: 1,       # Left Knee
        5: 2,       # Right Knee
        6: 3,       # Spine 2
        7: 4,       # Left Ankle
        8: 5,       # Right Ankle,
        9: 6,       # Spine 3
        10: 7,      # Left Foot
        11: 8,      # Right Foot
        12: 9,      # Neck
        13: 9,      # Left Collar
        14: 9,      # Right Collar
        15: 12,     # Head
        16: 13,     # Left Shoulder
        17: 14,     # Right Shoulder
        18: 16,     # Left Elbow
        19: 17,     # Right Elbow
        20: 18,     # Left Wrist
        21: 19,     # Right Wrist
        # Hand joints
        22: 20,     # Left Index 1
        23: 22,     # Left Index 2
        24: 23,     # Left Index 3
        25: 20,     # Left Middle 1
        26: 25,     # Left Middle 2
        27: 26,     # Left Middle 3
        28: 20,     # Left Pinky 1
        29: 28,     # Left Pinky 2
        30: 29,     # Left Pinky 3
        31: 20,     # Left Ring 1
        32: 31,     # Left Ring 2
        33: 32,     # Left Ring 3
        34: 20,     # Left Thumb 1
        35: 34,     # Left Thumb 2
        36: 35,     # Left Thumb 3
        37: 21,     # Right Index 1
        38: 37,     # Right Index 2
        39: 38,     # Right Index 3,
        40: 21,     # Right Middle 1
        41: 40,     # Right Middle 2
        42: 41,     # Right Middle 3
        43: 21,     # Right Pinky 1
        44: 43,     # Right Pinky 2
        45: 44,     # Right Pinky 3
        46: 21,     # Right Ring 1
        47: 46,     # Right Ring 2
        48: 47,     # Right Ring 3
        49: 21,     # Right Thumb 1
        50: 49,     # Right Thumb 2
        51: 50      # Right Thumb 3
    },
    # From SMPL-X blender addon's kinematic tree
    BodyModelType.SMPLX: {
        0: None,    # Pelvis
        # Body joints
        1: 0,       # Left Hip
        2: 0,       # Right Hip
        3: 0,       # Spine 1
        4: 1,       # Left Knee
        5: 2,       # Right Knee
        6: 3,       # Spine 2
        7: 4,       # Left Ankle
        8: 5,       # Right Ankle
        9: 6,       # Spine 3
        10: 7,      # Left Foot
        11: 8,      # Right Foot
        12: 9,      # Neck
        13: 9,      # Left Collar
        14: 9,      # Right Collar
        15: 12,     # Head
        16: 13,     # Left Shoulder
        17: 14,     # Right Shoulder
        18: 16,     # Left Elbow
        19: 17,     # Right Elbow
        20: 18,     # Left Wrist
        21: 19,     # Right Wrist
        # Face joints
        22: 15,     # Jaw
        23: 15,     # Left Eye
        24: 15,     # Right Eye
        # Hand joints
        25: 20,     # Left Index 1
        26: 25,     # Left Index 2
        27: 26,     # Left Index 3
        28: 20,     # Left Middle 1
        29: 28,     # Left Middle 2
        30: 29,     # Left Middle 3
        31: 20,     # Left Pinky 1
        32: 31,     # Left Pinky 2
        33: 32,     # Left Pinky 3
        34: 20,     # Left Ring 1
        35: 34,     # Left Ring 2
        36: 35,     # Left Ring 3
        37: 20,     # Left Thumb 1
        38: 37,     # Left Thumb 2
        39: 38,     # Left Thumb 3
        40: 21,     # Right Index 1
        41: 40,     # Right Index 2
        42: 41,     # Right Index 3
        43: 21,     # Right Middle 1
        44: 43,     # Right Middle 2
        45: 44,     # Right Middle 3
        46: 21,     # Right Pinky 1
        47: 46,     # Right Pinky 2
        48: 47,     # Right Pinky 3
        49: 21,     # Right Ring 1
        50: 49,     # Right Ring 2
        51: 50,     # Right Ring 3
        52: 21,     # Right Thumb 1
        53: 52,     # Right Thumb 2
        54: 53,     # Right Thumb 3
    },
    BodyModelType.SUPR: {
        0: None,    # Pelvis
        # Body joints
        1: 0,       # Left Hip
        2: 0,       # Right Hip
        3: 0,       # Spine 1
        4: 1,       # Left Knee
        5: 2,       # Right Knee
        6: 3,       # Spine 2
        7: 4,       # Left Ankle
        8: 5,       # Right Ankle
        9: 6,       # Spine 3
        10: 7,      # Left Ball of the foot
        11: 8,      # Right Ball of the foot
        12: 9,      # Neck
        13: 9,      # Left Collar
        14: 9,      # Right Collar
        15: 12,     # Head
        16: 13,     # Left Shoulder
        17: 14,     # Right Shoulder
        18: 16,     # Left Elbow
        19: 17,     # Right Elbow
        20: 18,     # Left Wrist
        21: 19,     # Right Wrist
        # Face joints
        22: 15,     # Jaw
        23: 15,     # Left Eye
        24: 15,     # Right Eye
        # Hand joints
        25: 20,     # Left Index 1
        26: 25,     # Left Index 2
        27: 26,     # Left Index 3
        28: 20,     # Left Middle 1
        29: 28,     # Left Middle 2
        30: 29,     # Left Middle 3
        31: 20,     # Left Pinky 1
        32: 31,     # Left Pinky 2
        33: 32,     # Left Pinky 3
        34: 20,     # Left Ring 1
        35: 34,     # Left Ring 2
        36: 35,     # Left Ring 3
        37: 20,     # Left Thumb 1
        38: 37,     # Left Thumb 2
        39: 38,     # Left Thumb 3
        40: 21,     # Right Index 1
        41: 40,     # Right Index 2
        42: 41,     # Right Index 3
        43: 21,     # Right Middle 1
        44: 43,     # Right Middle 2
        45: 44,     # Right Middle 3
        46: 21,     # Right Pinky 1
        47: 46,     # Right Pinky 2
        48: 47,     # Right Pinky 3
        49: 21,     # Right Ring 1
        50: 49,     # Right Ring 2
        51: 50,     # Right Ring 3
        52: 21,     # Right Thumb 1
        53: 52,     # Right Thumb 2
        54: 53,     # Right Thumb 3
        # Feet joints
        55: 10,     # Left Big Toe 1
        56: 55,     # Left Big Toe 2
        57: 10,     # Left Pointer Toe 1
        58: 57,     # Left Pointer Toe 2
        59: 10,     # Left Middle Toe 1
        60: 59,     # Left Middle Toe 2
        61: 10,     # Left Ring Toe 1
        62: 61,     # Left Ring Toe 2
        63: 10,     # Left Baby Toe 1
        64: 63,     # Left Baby Toe 2
        65: 11,     # Right Big Toe 1
        66: 65,     # Right Big Toe 2
        67: 11,     # Right Pointer Toe 1
        68: 67,     # Right Pointer Toe 2
        69: 11,     # Right Middle Toe 1
        70: 69,     # Right Middle Toe 2
        71: 11,     # Right Ring Toe 1
        72: 71,     # Right Ring Toe 2
        73: 11,     # Right Baby Toe 1
        74: 73,     # Right Baby Toe 2
    },
    BodyModelType.STAR: {
        0: None,    # Pelvis
        # Body joints
        1: 0,       # Left Hip
        2: 0,       # Right Hip
        3: 0,       # Spine 1
        4: 1,       # Left Knee
        5: 2,       # Right Knee
        6: 3,       # Spine 2
        7: 4,       # Left Ankle
        8: 5,       # Right Ankle,
        9: 6,       # Spine 3
        10: 7,      # Left Foot
        11: 8,      # Right Foot
        12: 9,      # Neck
        13: 9,      # Left Collar
        14: 9,      # Right Collar
        15: 12,     # Head
        16: 13,     # Left Shoulder
        17: 14,     # Right Shoulder
        18: 16,     # Left Ellbow
        19: 17,     # Right Ellbow
        20: 18,     # Left Wrist
        21: 19,     # Right Wrist
        22: 20,     # Left Hand
        23: 21      # Right Hand
    }
}
#CHILD_JOINT_MAPPINGS = {
#    model: {
#        joint_id: [x for x in PARENT_JOINT_MAPPINGS[model].keys() if PARENT_JOINT_MAPPINGS[model][x] == joint_id]
#        for joint_id in PARENT_JOINT_MAPPINGS[model].keys()
#    }
#    for model in PARENT_JOINT_MAPPINGS.keys()
#}
CHILD_JOINT_MAPPINGS = {
    BodyModelType.SMPL: {
        0: [1, 2, 3],
        1: [4],
        2: [5],
        3: [6],
        4: [7],
        5: [8],
        6: [9],
        7: [10],
        8: [11],
        9: [12, 13, 14],
        10: [],
        11: [],
        12: [15],
        13: [16],
        14: [17],
        15: [],
        16: [18],
        17: [19],
        18: [20],
        19: [21],
        20: [22],
        21: [23],
        22: [],
        23: []
    },
    BodyModelType.SMPLH: {
        0: [1, 2, 3],
        1: [4],
        2: [5],
        3: [6],
        4: [7],
        5: [8],
        6: [9],
        7: [10],
        8: [11],
        9: [12, 13, 14],
        10: [],
        11: [],
        12: [15],
        13: [16],
        14: [17],
        15: [],
        16: [18],
        17: [19],
        18: [20],
        19: [21],
        20: [22, 25, 28, 31, 34],
        21: [37, 40, 43, 46, 49],
        22: [23],
        23: [24],
        24: [],
        25: [26],
        26: [27],
        27: [],
        28: [29],
        29: [30],
        30: [],
        31: [32],
        32: [33],
        33: [],
        34: [35],
        35: [36],
        36: [],
        37: [38],
        38: [39],
        39: [],
        40: [41],
        41: [42],
        42: [],
        43: [44],
        44: [45],
        45: [],
        46: [47],
        47: [48],
        48: [],
        49: [50],
        50: [51],
        51: []
    },
    BodyModelType.SMPLX: {
        0: [1, 2, 3],
        1: [4],
        2: [5],
        3: [6],
        4: [7],
        5: [8],
        6: [9],
        7: [10],
        8: [11],
        9: [12, 13, 14],
        10: [],
        11: [],
        12: [15],
        13: [16],
        14: [17],
        15: [22, 23, 24],
        16: [18],
        17: [19],
        18: [20],
        19: [21],
        20: [25, 28, 31, 34, 37],
        21: [40, 43, 46, 49, 52],
        22: [],
        23: [],
        24: [],
        25: [26],
        26: [27],
        27: [],
        28: [29],
        29: [30],
        30: [],
        31: [32],
        32: [33],
        33: [],
        34: [35],
        35: [36],
        36: [],
        37: [38],
        38: [39],
        39: [],
        40: [41],
        41: [42],
        42: [],
        43: [44],
        44: [45],
        45: [],
        46: [47],
        47: [48],
        48: [],
        49: [50],
        50: [51],
        51: [],
        52: [53],
        53: [54],
        54: []
    },
    BodyModelType.SUPR: {
        0: [1, 2, 3],
        1: [4],
        2: [5],
        3: [6],
        4: [7],
        5: [8],
        6: [9],
        7: [10],
        8: [11],
        9: [12, 13, 14],
        10: [55, 57, 59, 61, 63],
        11: [65, 67, 69, 71, 73],
        12: [15],
        13: [16],
        14: [17],
        15: [22, 23, 24],
        16: [18],
        17: [19],
        18: [20],
        19: [21],
        20: [25, 28, 31, 34, 37],
        21: [40, 43, 46, 49, 52],
        22: [],
        23: [],
        24: [],
        25: [26],
        26: [27],
        27: [],
        28: [29],
        29: [30],
        30: [],
        31: [32],
        32: [33],
        33: [],
        34: [35],
        35: [36],
        36: [],
        37: [38],
        38: [39],
        39: [],
        40: [41],
        41: [42],
        42: [],
        43: [44],
        44: [45],
        45: [],
        46: [47],
        47: [48],
        48: [],
        49: [50],
        50: [51],
        51: [],
        52: [53],
        53: [54],
        54: [],
        55: [56],
        56: [],
        57: [58],
        58: [],
        59: [60],
        60: [],
        61: [62],
        62: [],
        63: [64],
        64: [],
        65: [66],
        66: [],
        67: [68],
        68: [],
        69: [70],
        70: [],
        71: [72],
        72: [],
        73: [74],
        74: []
    },
    BodyModelType.STAR: {
        0: [1, 2, 3],
        1: [4],
        2: [5],
        3: [6],
        4: [7],
        5: [8],
        6: [9],
        7: [10],
        8: [11],
        9: [12, 13, 14],
        10: [],
        11: [],
        12: [15],
        13: [16],
        14: [17],
        15: [],
        16: [18],
        17: [19],
        18: [20],
        19: [21],
        20: [22],
        21: [23],
        22: [],
        23: []
    }
}

# For converting hand joint indices between SMPL+H, SMPL-X, and SUPR
# Outer dict key: conv_to | inner dict key: conv_from
# e.g. for SMPL-X from SMPL+H -> -3
MODEL_HAND_JOINT_OFFSETS = {
    BodyModelType.SMPLH: {
        BodyModelType.SMPLH: 0,
        BodyModelType.SMPLX: 3,
        BodyModelType.SUPR: 3
    },
    BodyModelType.SMPLX: {
        BodyModelType.SMPLH: -3,
        BodyModelType.SMPLX: 0,
        BodyModelType.SUPR: 0
    },
    BodyModelType.SUPR: {
        BodyModelType.SMPLH: -3,
        BodyModelType.SMPLX: 0,
        BodyModelType.SUPR: 0
    }
}

JOINT_NAMES = {
    BodyModelType.SMPL: {
        0: 'pelvis',
        # Body joints
        1: 'left_hip',
        2: 'right_hip',
        3: 'spine_1',
        4: 'left_knee',
        5: 'right_knee',
        6: 'spine_2',
        7: 'left_ankle',
        8: 'right_ankle',
        9: 'spine_3',
        10: 'left_foot',
        11: 'right_foot',
        12: 'neck',
        13: 'left_collar',
        14: 'right_collar',
        15: 'head',
        16: 'left_shoulder',
        17: 'right_shoulder',
        18: 'left_ellbow',
        19: 'right_ellbow',
        20: 'left_wrist',
        21: 'right_wrist',
        22: 'left_hand',
        23: 'right_hand'
    },
    BodyModelType.SMPLH: {
        0: 'pelvis',
        # Body joints
        1: 'left_hip',
        2: 'right_hip',
        3: 'spine_1',
        4: 'left_knee',
        5: 'right_knee',
        6: 'spine_2',
        7: 'left_ankle',
        8: 'right_ankle',
        9: 'spine_3',
        10: 'left_foot',
        11: 'right_foot',
        12: 'neck',
        13: 'left_collar',
        14: 'right_collar',
        15: 'head',
        16: 'left_shoulder',
        17: 'right_shoulder',
        18: 'left_ellbow',
        19: 'right_ellbow',
        20: 'left_wrist',
        21: 'right_wrist',
        # Hand joints
        22: 'left_index_1',
        23: 'left_index_2',
        24: 'left_index_3',
        25: 'left_middle_1',
        26: 'left_middle_2',
        27: 'left_middle_3',
        28: 'left_pinky_1',
        29: 'left_pinky_2',
        30: 'left_pinky_3',
        31: 'left_ring_1',
        32: 'left_ring_2',
        33: 'left_ring_3',
        34: 'left_thumb_1',
        35: 'left_thumb_2',
        36: 'left_thumb_3',
        37: 'right_index_1',
        38: 'right_index_2',
        39: 'right_index_3',
        40: 'right_middle_1',
        41: 'right_middle_2',
        42: 'right_middle_3',
        43: 'right_pinky_1',
        44: 'right_pinky_2',
        45: 'right_pinky_3',
        46: 'right_ring_1',
        47: 'right_ring_2',
        48: 'right_ring_3',
        49: 'right_thumb_1',
        50: 'right_thumb_2',
        51: 'right_thumb_3'
    },
    BodyModelType.SMPLX: {
        0: 'pelvis',
        # Body joints
        1: 'left_hip',
        2: 'right_hip',
        3: 'spine_1',
        4: 'left_knee',
        5: 'right_knee',
        6: 'spine_2',
        7: 'left_ankle',
        8: 'right_ankle',
        9: 'spine_3',
        10: 'left_foot',
        11: 'right_foot',
        12: 'neck',
        13: 'left_collar',
        14: 'right_collar',
        15: 'head',
        16: 'left_shoulder',
        17: 'right_shoulder',
        18: 'left_ellbow',
        19: 'right_ellbow',
        20: 'left_wrist',
        21: 'right_wrist',
        # Face joints
        22: 'jaw',
        23: 'left_eye',
        24: 'eight_eye',
        # Hand joints
        25: 'left_index_1',
        26: 'left_index_2',
        27: 'left_index_3',
        28: 'left_middle_1',
        29: 'left_middle_2',
        30: 'left_middle_3',
        31: 'left_pinky_1',
        32: 'left_pinky_2',
        33: 'left_pinky_3',
        34: 'left_ring_1',
        35: 'left_ring_2',
        36: 'left_ring_3',
        37: 'left_thumb_1',
        38: 'left_thumb_2',
        39: 'left_thumb_3',
        40: 'right_index_1',
        41: 'right_index_2',
        42: 'right_index_3',
        43: 'right_middle_1',
        44: 'right_middle_2',
        45: 'right_middle_3',
        46: 'right_pinky_1',
        47: 'right_pinky_2',
        48: 'right_pinky_3',
        49: 'right_ring_1',
        50: 'right_ring_2',
        51: 'right_ring_3',
        52: 'right_thumb_1',
        53: 'right_thumb_2',
        54: 'right_thumb_3'
    },
    BodyModelType.SUPR: {
        0: 'pelvis',
        # Body joints
        1: 'left_hip',
        2: 'right_hip',
        3: 'spine_1',
        4: 'left_knee',
        5: 'right_knee',
        6: 'spine_2',
        7: 'left_ankle',
        8: 'right_ankle',
        9: 'spine_3',
        10: 'left_ball_of_the_foot',
        11: 'right_ball_of_the_foot',
        12: 'neck',
        13: 'left_collar',
        14: 'right_collar',
        15: 'head',
        16: 'left_shoulder',
        17: 'right_shoulder',
        18: 'left_ellbow',
        19: 'right_ellbow',
        20: 'left_wrist',
        21: 'right_wrist',
        # Face joints
        22: 'jaw',
        23: 'left_eye',
        24: 'eight_eye',
        # Hand joints
        25: 'left_index_1',
        26: 'left_index_2',
        27: 'left_index_3',
        28: 'left_middle_1',
        29: 'left_middle_2',
        30: 'left_middle_3',
        31: 'left_pinky_1',
        32: 'left_pinky_2',
        33: 'left_pinky_3',
        34: 'left_ring_1',
        35: 'left_ring_2',
        36: 'left_ring_3',
        37: 'left_thumb_1',
        38: 'left_thumb_2',
        39: 'left_thumb_3',
        40: 'right_index_1',
        41: 'right_index_2',
        42: 'right_index_3',
        43: 'right_middle_1',
        44: 'right_middle_2',
        45: 'right_middle_3',
        46: 'right_pinky_1',
        47: 'right_pinky_2',
        48: 'right_pinky_3',
        49: 'right_ring_1',
        50: 'right_ring_2',
        51: 'right_ring_3',
        52: 'right_thumb_1',
        53: 'right_thumb_2',
        54: 'right_thumb_3',
        # Feet joints
        55: 'left_big_toe_1',
        56: 'left_big_toe_2',
        57: 'left_pointer_toe_1',
        58: 'left_pointer_toe_2',
        59: 'left_middle_toe_1',
        60: 'left_middle_toe_2',
        61: 'left_ring_toe_1',
        62: 'left_ring_toe_2',
        63: 'left_baby_toe_1',
        64: 'left_baby_toe_2',
        65: 'right_big_toe_1',
        66: 'right_big_toe_2',
        67: 'right_pointer_toe_1',
        68: 'right_pointer_toe_2',
        69: 'right_middle_toe_1',
        70: 'right_middle_toe_2',
        71: 'right_ring_toe_1',
        72: 'right_ring_toe_2',
        73: 'right_baby_toe_1',
        74: 'right_baby_toe_2'
    },
    BodyModelType.STAR: {
        0: 'pelvis',
        # Body joints
        1: 'left_hip',
        2: 'right_hip',
        3: 'spine_1',
        4: 'left_knee',
        5: 'right_knee',
        6: 'spine_2',
        7: 'left_ankle',
        8: 'right_ankle',
        9: 'spine_3',
        10: 'left_foot',
        11: 'right_foot',
        12: 'neck',
        13: 'left_collar',
        14: 'right_collar',
        15: 'head',
        16: 'left_shoulder',
        17: 'right_shoulder',
        18: 'left_ellbow',
        19: 'right_ellbow',
        20: 'left_wrist',
        21: 'right_wrist',
        22: 'left_hand',
        23: 'right_hand'
    }
}
