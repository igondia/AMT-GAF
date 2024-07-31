# --------------------------------------------------------
# AMT-GAF
# Licensed under BSD-2 License
# Written by Iván González
# --------------------------------------------------------

"""AMT-GAF config system.

This file specifies default config options for AMT-GAF. You should not
change values in this file. Instead, you should write a config file (in yaml)
and use cfg_from_file(yaml_file) to load it and override the default options.
"""

import os
import os.path as osp
import numpy as np
import pdb
# `pip install easydict` if you don't have it
from easydict import EasyDict as edict

__C = edict()
# Consumers can get config by:
#   from active_object_detection import cfg
cfg = __C

#Type of Parametrization
__C.PARAMETRIZATION = 0


#Feature input weights
__C.INPUT_WEIGHTS = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],dtype=np.float32)
__C.FMEANS = np.array([8.8672e+01,  1.3207e+01,  9.2209e+00,  1.9000e-01, -1.2246e+01,
        5.8894e-04,  7.5189e-04,  3.9385e-02])
__C.FSTDS = np.array([7.0142e+01, 9.0292e+01, 8.9751e+01, 3.1135e-01, 4.6109e+00,
       4.5223e-02, 2.6700e-02, 5.8326e-02])


#Frame step
__C.F_S = 1
__C.TIME_MARKS = np.array([5, 10, 15, 20,25,30,35,40,45,50,200])

#Other parameters related to dynamic features
__C.MAXMV=100.0
__C.TH_MOV = 5.0
#Score for the Background class 
__C.BG_SCORE = 0.0

# Automatic normalization (for invisible dataset)
__C.AUTOMATIC_NORM = False

#Frames to predict
__C.FRS_TO_PREDICT = 1

#If set, we leave some frames from the initial gaze 
__C.PRE_FRAMES = 0

#Grid size for the spatial maps
__C.GRID_SIZE = np.array([8, 15],dtype=int)
#Sigma of spatial maps
__C.SIGMA_MAPS = 10.0
__C.SIGMA_MAPS1 = 105.0

#Factor of object working memory
__C.MEMORY_FACTOR = 0.9
#Threshold to decide that we have identified an object 
__C.TH_VWM = 0.10

# Architecture Parameters
__C.ATT_SIZE  = 256 #Hidden Size for Attention
__C.NHEAD = 1
__C.ZLENGTH = 256
__C.ACT_LAYERS_SIZE  = 256
__C.MAP_OP = 'sum'
__C.DROPOUT = 0.0
__C.TEMPORAL_HORIZON = 200

# Root directory of project
__C.ROOT_DIR = osp.abspath(osp.join(osp.dirname(__file__), '..', '..'))
__C.DB_DIR = osp.abspath(osp.join(osp.dirname(__file__), '..', '..','Data','Datasets','GITW'))
__C.CACHE_DIR = osp.abspath(osp.join(__C.ROOT_DIR,'data','cache')) 

#Random seed
__C.SEED = 999

# Place outputs under an experiments directory
__C.EXP_DIR = 'default'

__C.OBJ_PIXEL_MEANS = np.array([0.485, 0.456, 0.406])
__C.OBJ_PIXEL_STDS = np.array([0.229, 0.224, 0.225])
__C.POS_ENCODING_TYPE = 'sum'
__C.POS_ENCODING_LENGTH = 0

# Last Frames where the active object should be dominant to consider a sequence as valuid
__C.FRAMES_ACTIVE_OBJ  = 10

#Training options
__C.TRAIN = edict()

# Object permutation
__C.TRAIN.PERMUTE_OBJ  = False

# random offset for frames
__C.TRAIN.NOISE_FR_IDX = False
# Samples to use per minibatch
__C.TRAIN.SAMPLES_PER_BATCH  = 1

# Minibatch size (number of regions of interest [ROIs])
__C.TRAIN.WEIGHTED_SAMPLER = False
__C.TRAIN.WEIGHTED_SAMPLER_PARAM = 1.0
__C.TRAIN.BATCH_SIZE = 180
# Action LOSS parameters
# If positive, we learn the action only the pre_frames before it finishes, if 0-negative, simply requires t>=ACTLOSS_MIN_FRAMES_DET
__C.TRAIN.ACTLOSS_PRE_FRAMES = 25
#Minimum number of frames to establish a detection
__C.TRAIN.ACTLOSS_MIN_FRAMES_DET = 5
#Multiplier to weight sequence parts => before fixation, after fixation
__C.TRAIN.ACTLOSS_MUL = 2.0
#Parameters for Assymetric Multi-task Loss 
__C.TRAIN.LOSS_WEIGHTS =  np.array([1.0, 1.0])
__C.TRAIN.LOSS_TH = 1.0
__C.TRAIN.LOSS_ETA = 1.0
__C.TRAIN.LOSS_MAX_ITERS = 2

#Accuracy Computation PAarameters
__C.TRAIN.ACC_MIN_FRAMES_DET = 3
__C.TRAIN.ACC_FRAMES_BETWEEN = 5
__C.TRAIN.ACC_PRE_FRAMES = 75

# Iterations between snapshots
__C.TRAIN.SNAPSHOT_ITERS = 10000

#Learning parameters
__C.TRAIN.LR = 1e-5
__C.TRAIN.MOMENTUM = 0.9
__C.TRAIN.BETA1 = 0.9
__C.TRAIN.BETA2 = 0.999
__C.TRAIN.WD = 0.0005
__C.TRAIN.LR_STEP_SIZE = 200
__C.TRAIN.LR_GAMMA = 0.5
__C.TRAIN.NUM_WORKERS = 4
__C.TRAIN.CLIP_GRADS = 10.0
__C.TRAIN.NOISE_STD = 0.1


# Testing options
#

__C.TEST = edict()
__C.TEST.DETECTION_WINDOW = 1.0
__C.TEST.DETECTION_GUARD = 0.0
__C.TEST.PRE_FRAMES = 0
__C.TEST.DETECTION_TH = 0.38 
__C.TEST.NUM_WORKERS = 4
__C.TEST.SAVE_GAZE_PATTERNS = False

__C.TEST.FR_TO_SHOW = 0 #0 The frame to show intest


def get_output_dir(imdb, net=None):
    """Return the directory where experimental artifacts are placed.

    A canonical path is built using the name from an imdb and a network
    (if not None).
    """
    path = osp.abspath(osp.join(__C.ROOT_DIR, 'output', __C.EXP_DIR, imdb.name))
    if net is None:
        return path
    else:
        return osp.join(path, net)

def _merge_a_into_b(a, b):
    """Merge config dictionary a into config dictionary b, clobbering the
    options in b whenever they are also specified in a.
    """
    if type(a) is not edict:
        return

    for k, v in a.items():
        # a must specify keys that are in b
        if k not in b:
            raise KeyError('{} is not a valid config key'.format(k))

        # the types must match, too
        if type(b[k]) is not type(v):
            v=np.array(v,dtype=b[k].dtype);
            if type(b[k]) is not type(v):
                raise ValueError(('Type mismatch ({} vs. {}) '
                              'for config key: {}').format(type(b[k]),
                                                           type(v), k))

        # recursively merge dicts
        if type(v) is edict:
            try:
                _merge_a_into_b(a[k], b[k])
            except:
                print('Error under config key: {}'.format(k))
                raise
        else:
            b[k] = v

def cfg_from_file(filename):
    """Load a config file and merge it into the default options."""
    import yaml
    with open(filename, 'r') as f:
        yaml_cfg = edict(yaml.load(f,Loader=yaml.FullLoader))

    _merge_a_into_b(yaml_cfg, __C)

def cfg_from_list(cfg_list):
    """Set config keys via list (e.g., from command line)."""
    from ast import literal_eval
    assert len(cfg_list) % 2 == 0
    for k, v in zip(cfg_list[0::2], cfg_list[1::2]):
        key_list = k.split('.')
        d = __C
        for subkey in key_list[:-1]:
            assert d.has_key(subkey)
            d = d[subkey]
        subkey = key_list[-1]
        assert d.has_key(subkey)
        try:
            value = literal_eval(v)
        except:
            # handle the case when v is a string literal
            value = v
        assert type(value) == type(d[subkey]), \
            'type {} does not match original type {}'.format(
            type(value), type(d[subkey]))
        d[subkey] = value
