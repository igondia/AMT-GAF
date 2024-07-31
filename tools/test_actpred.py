# --------------------------------------------------------
# AMT-GAF
# Licensed under BSD-2 License
# Written by Iván González
# --------------------------------------------------------

"""Test a Fast R-CNN network on an image database."""
import pdb
import numpy as np
import _init_paths
from sequence_prediction.config import cfg, cfg_from_file, cfg_from_list
from datasets.factory import get_imdb
import argparse
import pprint
import time, os, sys

import torch
import numpy.random as npr
import random

np.finfo(np.dtype("float32"))
np.finfo(np.dtype("float64"))

def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Test a Fast R-CNN network')
    parser.add_argument('--net', dest='trained_net',
                        help='path to net weights',
                        default=None, type=str)
    parser.add_argument('--cfg', dest='cfg_file',
                        help='optional config file', default=None, type=str)
    parser.add_argument('--wait', dest='wait',
                        help='wait until net file exists',
                        default=True, type=bool)
    parser.add_argument('--imdb', dest='imdb_name',
                        help='dataset to test',
                        default='gitw_test', type=str)
    parser.add_argument('--seqdb', dest='seqdb_name',
                        help='dataset w',
                        default='gitw_test_fix', type=str)                
    parser.add_argument('--set', dest='set_cfgs',
                        help='set config keys', default=None,
                        nargs=argparse.REMAINDER)

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    
    args = parse_args()

    print('Called with args:')
    print(args)
    
    if args.cfg_file is not None:
        cfg_from_file(args.cfg_file)
    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs)

    
    
    print('Using config:')
    pprint.pprint(cfg)

    manualSeed = cfg.SEED 
    print('Random Seed: %d'%manualSeed)
    npr.seed(manualSeed)
    random.seed(manualSeed)
    torch.manual_seed(manualSeed)
    # torch uses some non-deterministic algorithms
    torch.backends.cudnn.enabled = False
    
    while not os.path.exists(args.trained_net) and args.wait:
        print('Waiting for {} to exist...'.format(args.trained_net))
        time.sleep(10)
    
    imdb = get_imdb(args.imdb_name,cfg.DB_DIR)
    #Set the bb name
    imdb.set_seq_db(args.seqdb_name)
    
   
    with torch.no_grad():
        if "gitw" in args.imdb_name:
            from sequence_prediction.test_action_forecasting import test_net
            test_net(args.trained_net, imdb)
        elif "sharon" in args.imdb_name:
            from sequence_prediction.test_action_forecasting import test_net
            test_net(args.trained_net, imdb)
        elif "invisible" in args.imdb_name:
            from sequence_prediction.test_intention_prediction import test_net
            test_net(args.trained_net, imdb)
        