# --------------------------------------------------------
# AMT-GAF
# Licensed under BSD-2 License
# Written by Iván González
# --------------------------------------------------------

"""Factory method for easily getting imdbs by name."""

__sets = {}
import pdb
from datasets.grasping_in_the_wild import grasping_in_the_wild 
from datasets.sharon import sharon
from datasets.invisible import invisible

import numpy as np





db_dir=None
           
for fold in np.arange(1,10,1):            
    for split in ['train','trainf', 'test']:
        name = 'gitw_{}_{:d}'.format(split,fold)
        __sets[name] = (lambda split=split, fold=fold,db_dir=db_dir:
                grasping_in_the_wild(split, fold,db_dir=db_dir))
                
            
for split in ['train', 'val', 'trainval', 'test']:
    name = 'sharon_{}'.format(split)
    __sets[name] = (lambda split=split, db_dir=db_dir:
                sharon(split,fold=0,db_dir=db_dir))
     
for fold in np.arange(1,10,1):            
    for split in ['train','test']:
        name = 'sharon_{}_{:d}'.format(split,fold)
        __sets[name] = (lambda split=split, fold=fold,db_dir=db_dir:
                sharon(split, fold,db_dir=db_dir))
            
for fold in np.arange(1,10,1):            
    for split in ['train','test']:
        name = 'invisible_{}_{:d}'.format(split,fold)
        __sets[name] = (lambda split=split, fold=fold,db_dir=db_dir:
                invisible(split, fold,db_dir=db_dir))

            
def get_imdb(name,db_dir=None):
    
    """Get an imdb (image database) by name."""
    if name not in __sets:
        raise KeyError('Unknown dataset: {}'.format(name))
    return __sets[name](db_dir=db_dir)

def list_imdbs():
    """List all registered imdbs."""
    return __sets.keys()
