# --------------------------------------------------------
# AMT-GAF
# Licensed under BSD-2 License
# Written by Iván González
# --------------------------------------------------------

import pdb
from seq_data_layer.video_features import get_video_features
import torch
from torch.utils.data import Dataset
import numpy as np
import lmdb
import copy
from sequence_prediction.config import cfg

class SeqDataset(Dataset):
    """Sequence Dataset"""

    def __init__(self, seqdb,act_classes,verb_classes,obj_classes,act_class_probs=None,cfgPath=None,test=False):
        # seqdb['video_data']=seqdb['video_data'][::25]#
        self.seqdb = seqdb 
        self.act_classes =act_classes
        self.verb_classes =verb_classes
        self.obj_classes =obj_classes
        
        self.num_actclasses = len(act_classes)
        self.num_verbclasses = len(verb_classes)
        self.num_objclasses = len(obj_classes)
        
        self.act_class_probs = act_class_probs
        
            
        self.cfgPath=cfgPath;
        self.test=test
        self.sample_weights=None
        
        #Initialize 
        self.fmeans=np.array(cfg.FMEANS)
        self.fstds=np.array(cfg.FSTDS)
        
        
    def __len__(self):
        return len(self.seqdb['video_data'])
    
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        #Obtain the item
        samples=get_video_features(self.seqdb['video_data'][idx],
                              self.fmeans,self.fstds,
                              cfgPath=self.cfgPath,
                              test=self.test)
        samples['video_idx']=idx;
        return samples



    def compare_models(self, model_1, model_2):
        
        models_differ = 0
        for key_item_1, key_item_2 in zip(model_1.state_dict().items(), model_2.state_dict().items()):
            if torch.equal(key_item_1[1], key_item_2[1]):
                pass
            else:
                models_differ += 1
                if (key_item_1[0] == key_item_2[0]):
                    print('Mismtach found at', key_item_1[0])
                else:
                    raise Exception
        if models_differ == 0:
            print('Models match perfectly! :)')

    def setSampleWeights(self,weights,vidx):
        if self.sample_weights is None:
            self.sample_weights = 10*np.ones((len(self.seqdb['video_data']),),dtype=np.float32)
            
   
        self.sample_weights[vidx]=weights
        
    
    def getSampleWeights(self,vidx):
        if self.sample_weights is None:
            self.sample_weights = 10*np.ones((len(self.seqdb['video_data']),),dtype=np.float32)
        
        sample_weights=self.sample_weights[vidx]
        sample_weights=sample_weights/(sample_weights.mean()+1e-10)
        print(sample_weights)
        return sample_weights

    def setNormalization(self,fmeans,fstds):
        print('Set normalization values')
        self.fstds=fstds.copy()
        self.fmeans=fmeans.copy()
        
        