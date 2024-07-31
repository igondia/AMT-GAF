# --------------------------------------------------------
# AMT-GAF
# Licensed under BSD-2 License
# Written by Iván González
# --------------------------------------------------------

"""Train a Fast R-CNN network on a region of interest database."""

import _init_paths
from seq_data_layer.seqdb import get_training_seqdb
from sequence_prediction.config import cfg, cfg_from_file, cfg_from_list, get_output_dir
from datasets.factory import get_imdb
from seq_data_layer.SeqDataset import SeqDataset

import argparse
import pprint
import numpy as np
import sys
import pdb
import random
import numpy.random as npr
import torch


def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Train an action prediction model')
    parser.add_argument('--epochs', dest='epochs',
                        help='number of epochs to train',
                        default=25, type=int)
    parser.add_argument('--pretrained', dest='pretrained_model',
                        help='initialize with pretrained model weights',
                        default=True, type=bool)
    parser.add_argument('--cfg', dest='cfg_file',
                        help='optional config file',
                        default=None, type=str)
    parser.add_argument('--imdb', dest='imdb_name',
                        help='dataset to train on',
                        default='gitw_train', type=str)
    parser.add_argument('--seqdb', dest='seqdb_name',
                        help='dataset identifier',
                        default='gitw_train', type=str)                        
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

    # Set random seed for reproducibility
    manualSeed = cfg.SEED;
    
    print('Random Seed: %d'%manualSeed)
    npr.seed(manualSeed)
    random.seed(manualSeed)
    torch.manual_seed(manualSeed)
    # torch uses some non-deterministic algorithms
    torch.backends.cudnn.enabled = False
    
    print('Using config:')
    pprint.pprint(cfg)


    imdb = get_imdb(args.imdb_name,db_dir=cfg.DB_DIR)
    
    #Set the bb name
    imdb.set_seq_db(args.seqdb_name)
    
    
    print('Loaded dataset `{:s}` for training'.format(imdb.name))
    seqdb = get_training_seqdb(imdb)
    
    #Convert roidb to a ROIDataset that can be decoded by Pytorch
    seqDataset = SeqDataset(seqdb,imdb.actclasses,imdb.verbclasses,imdb.objclasses,cfgPath=args.cfg_file)
    
    
    output_dir = get_output_dir(imdb, None)
    print('Output will be saved to `{:s}`'.format(output_dir))
    
    #Depending on the dataset, choose the right training function
    if "gitw" in args.imdb_name:
        from sequence_prediction.train_action_forecasting import train_net
        train_net(seqDataset, output_dir,pretrained_model=args.pretrained_model,epochs=args.epochs)
    elif "sharon" in args.imdb_name:
        from sequence_prediction.train_action_forecasting import train_net
        train_net(seqDataset, output_dir,pretrained_model=args.pretrained_model,epochs=args.epochs)
    elif "invisible" in args.imdb_name:
        from sequence_prediction.train_intention_prediction import train_net
        train_net(seqDataset, output_dir,pretrained_model=args.pretrained_model,epochs=args.epochs)
    
    print('End of training process')
