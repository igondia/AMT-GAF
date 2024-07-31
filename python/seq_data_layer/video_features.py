# --------------------------------------------------------
# AMT-GAF
# Licensed under BSD-2 License
# Written by Iván González
# --------------------------------------------------------

"""Compute minibatch blobs for training a Fast R-CNN network."""
import pdb
import numpy as np
import numpy.random as npr
from sequence_prediction.config import cfg
from scipy.special import softmax
from seq_data_layer.gazeFeatures import computeGazeFeatures


    
    
"""Function that reads video features
    video: video object
    fmeans: feature means for standardization
    fstds: feature stds for standardization
    obj_perm_idx: idxs for object permutation (data augmentation)
    VWM: Visual Working Memory
    cfgPath: path to config file
    test: True if we are on test
"""
def get_video_features(video,fmeans,fstds,
                  obj_perm_idx=None,VWM=None,cfgPath=None,test=False):
    
        
    from sequence_prediction.config import cfg, cfg_from_file 
    if cfgPath!=None:
        cfg_from_file(cfgPath)
   
    #GITW: Read pre-computed features (RGB, Flow, OBJ) and load precomputed gaze    
    if cfg.PARAMETRIZATION==1:
        sample= minibatch1(video,fmeans,fstds,obj_perm_idx,test=test)     
    #SHARON: Read pre-computed features (RGB, Flow, OBJ) and load precomputed gaze    
    elif cfg.PARAMETRIZATION==2:
        sample= minibatch2(video,fmeans,fstds,obj_perm_idx,test=test)         
    #INVISIBLE: Read pre-computed features (RGB, Flow, OBJ) and load precomputed gaze    
    elif cfg.PARAMETRIZATION==3:
        sample= minibatch3(video,fmeans,fstds,obj_perm_idx,test=test)         
    
    return sample



"""Function that implements, for GITW dataset, a minibatch for multi-task with VWM both in gaze-pred and action forecasting"""
def minibatch1(video,fmeans,fstds,obj_perm_idx=None,test=False):

    num_classes = video['data']['scores'].shape[1]
    nFrV=video['data']['fix'].shape[0]
    
    #Now we set the training frames depending on anticipation and observation times
    step=cfg.F_S
    
    if test==False:
        idxFr=np.arange(0,nFrV)
        frames=np.arange(0,nFrV,step)
        if cfg.TRAIN.BATCH_SIZE>1:
            nFr=cfg.TRAIN.SAMPLES_PER_BATCH
        else:
            nFr=nFrV
    else:
        idxFr=np.arange(0,nFrV)
        frames=np.arange(0,nFrV,step)
        nFr=nFrV
        
    labels = np.zeros((nFrV,), dtype=np.int64)
    time_blob = np.zeros((nFrV,), dtype=np.float32)
    labels[...]=(video['label']*video['data']['labels'][idxFr]).astype(int)[:,0];
    #Set time blob
    time_blob=np.cumsum((video['data']['labels'][idxFr]>0).astype(float))
    
    #Giving size to inputs    
    video_name=video['video'].split('/')[-1]
    
    
    img_tmpl = "Frame_{:d}"
    frames = np.array(list(map(lambda x: video_name+"/"+img_tmpl.format(x), idxFr)))
    
    
    if nFrV > nFr:
        frames=frames[-nFr:]
        idxFr=idxFr[-nFr:]
        labels=labels[-nFr:]
        time_blob=time_blob[-nFr:]
        nFrV=nFr
        idxFr=np.arange(0,nFrV)
        
    active_obj=labels.max()
    
    if test==False and cfg.TRAIN.PERMUTE_OBJ:
        active_obj=labels.max()
        #Option 1: permute everything except the background
        idx_perm=np.random.permutation(num_classes-1);
        idx_perm_0=np.concatenate(([0],idx_perm+1))
        idx_active_obj=np.nonzero(idx_perm_0==active_obj)[0][0];
        labels[np.nonzero(labels)[0]]=idx_active_obj
        
    else:
        idx_perm_0=[]
        
    #Read Gaze features
    data,gcoord_blob,smap_blob,scores_blob,flimits=computeGazeFeatures(video,idxFr,step,fmeans,fstds,idx_perm=idx_perm_0)
    
    #Padding with zeros at the end
    if test==False and nFrV<nFr:
        num_zeros = nFr-nFrV
        data=np.pad(data,((0, num_zeros),(0,0)));
        labels=np.pad(labels,(0,num_zeros),constant_values=-1);
        time_blob=np.pad(time_blob,(0,num_zeros),constant_values=-1);
        scores_blob=np.pad(scores_blob,((0, num_zeros),(0,0)));
        gcoord_blob=np.pad(gcoord_blob,((0, num_zeros),(0,0)));
        smap_blob=np.pad(smap_blob,((0, num_zeros),(0,0),(0,0),(0,0)));
    
    #Generate the sample
    sample={'data': data,'labels': labels, 'time' : time_blob, 'gaze': gcoord_blob, 
             'smap': smap_blob, 'scores': scores_blob, 'seq_length': nFrV}
    
    return sample

"""Function that implements, for SHARON dataset, a minibatch for multi-task with VWM both in gaze-pred and action forecasting"""
def minibatch2(video,fmeans,fstds,obj_perm_idx=None,test=False):
    """Given a seqdb, construct a minibatch sampled from it."""
    
    num_classes = video['data']['scores'].shape[1]
    
    #The number of frames is the last frame of the video
    nFrV=np.nonzero(video['data']['labels'])[0][-1]+1
    #Now we set the training frames depending on anticipation and observation times
    step=cfg.F_S
    if test==False:
        idxFr=np.arange(0,nFrV)
        frames=np.arange(0,nFrV,step)
        if cfg.TRAIN.BATCH_SIZE>1:
            nFr=cfg.TRAIN.SAMPLES_PER_BATCH
        else:
            nFr=nFrV
    else:
        idxFr=np.arange(0,nFrV)
        frames=np.arange(0,nFrV,step)
        nFr=nFrV
        
    labels = np.zeros((nFrV,), dtype=np.int64)
    time_blob = np.zeros((nFrV,), dtype=np.float32)
    
    #Get the labels
    labels[...]=video['data']['labels'][idxFr,0];
        
    #Set time blob
    time_blob=np.cumsum((labels>0).astype(float))
    
    
    #Giving size to inputs    
    video_name=video['video'].split('/')[-1]
    
    
    #b'P07-R01-PastaSalad/Frame_49470'
    img_tmpl = "Frame_{:d}"
    frames = np.array(list(map(lambda x: video_name+"/"+img_tmpl.format(x), idxFr)))
    
    
    if nFrV > nFr:
        frames=frames[-nFr:]
        idxFr=idxFr[-nFr:]
        labels=labels[-nFr:]
        time_blob=time_blob[-nFr:]
        nFrV=nFr
        idxFr=np.arange(0,nFrV)
    
    
    #Read data
    fixStart=np.nonzero(labels)[0][0]
       
            
    #By default we look into the last 10 frames to check if the label object is the most fixated one
    if cfg.FRAMES_ACTIVE_OBJ>0 and not test:
        active_obj=labels.max()
        fixStart=np.nonzero(labels)[0][0]
        fixEnd=np.nonzero(labels)[0][-1]
        
        #Compute proportion of GT against most fixated object (without considering background)
        #Option 1: accumulate scores
        ao_probs=softmax(video['data']['scores'][fixEnd-cfg.FRAMES_ACTIVE_OBJ:fixEnd+1,:],axis=1)
        ao_probs[:,0]=0
        ascores=ao_probs.sum(axis=0)
        best_class=np.argmax(ascores);
        prop=ascores[active_obj]/ascores[best_class]
        
        #If we are close, we can try to fix the scores to set the video as valid for training
        if(prop>0.8):
            inc=np.mean(video['data']['scores'][fixEnd-cfg.FRAMES_ACTIVE_OBJ:,best_class]-video['data']['scores'][fixEnd-cfg.FRAMES_ACTIVE_OBJ:,active_obj])
            video['data']['scores'][fixStart:,best_class]-=inc
        elif(best_class != active_obj): 
            labels[labels>0]=-1;
            time_blob[labels>0]=-1

        active_obj=labels.max()
             
    if test==False and cfg.TRAIN.PERMUTE_OBJ and active_obj>0:
        #Permute everything except the background
        idx_perm=np.random.permutation(num_classes-1);
        idx_perm_0=np.concatenate(([0],idx_perm+1))
        idx_active_obj=np.nonzero(idx_perm_0==active_obj)[0][0];
        labels[np.nonzero(labels)[0]]=idx_active_obj
        
        
    else:
        idx_active_obj=labels.max()
        idx_perm_0=[]
    
    
    
    #Read Gaze features
    data,gcoord_blob,smap_blob,scores_blob,flimits=computeGazeFeatures(video,idxFr,step,fmeans,fstds,idx_perm=idx_perm_0)
      
            
    # Padding with zeros at the end
    if test==False and nFrV<nFr:
        num_zeros = nFr-nFrV
        data=np.pad(data,((0, num_zeros),(0,0)));
        labels=np.pad(labels,(0,num_zeros),constant_values=-1);
        time_blob=np.pad(time_blob,(0,num_zeros),constant_values=-1);
        scores_blob=np.pad(scores_blob,((0, num_zeros),(0,0)));
        gcoord_blob=np.pad(gcoord_blob,((0, num_zeros),(0,0)));
        smap_blob=np.pad(smap_blob,((0, num_zeros),(0,0),(0,0),(0,0)));
        
    sample={'data': data,'labels': labels, 'time' : time_blob, 'gaze': gcoord_blob, 
             'smap': smap_blob, 'scores': scores_blob, 'seq_length': nFrV}
    
    return sample

"""Function that implements, for Invisible dataset, a minibatch for multi-task with VWM both in gaze-pred and action forecasting"""
def minibatch3(video,fmeans,fstds,obj_perm_idx=None,test=False):
    
    """Given a seqdb, construct a minibatch sampled from it."""
    
    num_obj_classes = video['data']['scores'].shape[1]
    #The number of frames is the last frame with labels in the the video (remove once objects are grasped)
    nFrV=np.nonzero(video['data']['labels']>=0)[0][-1]+1
    
    #Now we set the training frames depending on anticipation and observation times
    step=cfg.F_S
    if test==False:
        idxFr=np.arange(0,nFrV)
        frames=np.arange(0,nFrV,step)
        if cfg.TRAIN.BATCH_SIZE>1:
            nFr=cfg.TRAIN.SAMPLES_PER_BATCH
        else:
            nFr=nFrV
    else:
        idxFr=np.arange(0,nFrV)
        frames=np.arange(0,nFrV,step)
        nFr=nFrV
    labels = np.zeros((nFrV,), dtype=np.int64)
    time_blob = np.zeros((nFrV,), dtype=np.float32)
    
    #Get the labels
    
    labels[...]=video['data']['labels'][idxFr,0];
        
    #Set time blob
    time_blob=np.cumsum((labels>=0).astype(float))
    
    
    #Giving size to inputs    
    video_name=video['video'].split('/')[-1]
    
    
    #b'P07-R01-PastaSalad/Frame_49470'
    img_tmpl = "Frame_{:d}"
    frames = np.array(list(map(lambda x: video_name+"/"+img_tmpl.format(x), idxFr)))
    
    
    if nFrV > nFr:
        frames=frames[-nFr:]
        idxFr=idxFr[-nFr:]
        labels=labels[-nFr:]
        time_blob=time_blob[-nFr:]
        nFrV=nFr
        idxFr=np.arange(0,nFrV)
    
    
    
        
    label_action=labels.max()
    active_obj=(label_action+1)//2
    action=(label_action+1)%2
        
    if test==False and cfg.TRAIN.PERMUTE_OBJ and active_obj>0:
            
        #Option 1: permute everything except the background
        idx_perm=np.random.permutation(num_obj_classes-1);
        idx_perm_0=np.concatenate(([0],idx_perm+1))
        idx_active_obj=np.nonzero(idx_perm_0==active_obj)[0][0];
        labels[np.nonzero(labels)[0]]=2*idx_active_obj+action-1
            
            
    else:
        idx_active_obj=labels.max()
        idx_perm_0=[]
            
        #Read Gaze features
        data,gcoord_blob,smap_blob,scores_blob,flimits=computeGazeFeatures(video,idxFr,step,fmeans,fstds,idx_perm=idx_perm_0)
        
        
    # Padding with zeros at the end
    if test==False and nFrV<nFr:
        num_zeros = nFr-nFrV
        data=np.pad(data,((0, num_zeros),(0,0)));
        labels=np.pad(labels,(0,num_zeros),constant_values=-1);
        time_blob=np.pad(time_blob,(0,num_zeros),constant_values=-1);
        scores_blob=np.pad(scores_blob,((0, num_zeros),(0,0)));
        gcoord_blob=np.pad(gcoord_blob,((0, num_zeros),(0,0)));
        smap_blob=np.pad(smap_blob,((0, num_zeros),(0,0),(0,0),(0,0)));
        
    sample={'data': data, 'labels': labels, 'time' : time_blob, 'gaze': gcoord_blob, 
             'smap': smap_blob, 'scores': scores_blob, 'seq_length': nFrV, 'flimits': flimits}
    
    return sample

