# Source Generated with Decompyle++
# File: test_vae.pyc (Python 2.7)

'''Test a Fast R-CNN network on an imdb (image database).'''
import pdb
import warnings
warnings.filterwarnings("ignore")

from sequence_prediction.config import cfg, get_output_dir
import seq_data_layer.seqdb as sdl_seqdb
from utils.timer import Timer
import numpy as np
import os

import torch
from seq_data_layer.SeqDataset import SeqDataset
from torch.utils.data import DataLoader
from torch.nn import functional as F
import torch.nn as nn
import seqmodels.amt_gaf as sq
from seq_data_layer.normalization import computeNorms,updateMoms

def test_net(trained_model, imdb):
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    
    
    '''Test a Fast R-CNN network on an video database.'''
    output_dir = get_output_dir(imdb)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    _t = {
        'fr_decision': Timer(),
        'eval': Timer() }
    
    sdl_seqdb.prepare_seqdb(imdb)
    seqdb = imdb.seqdb
    
    """Build the dataset"""
    seqDataset = SeqDataset(seqdb,imdb.actclasses,imdb.actclasses,imdb.objclasses,test=True)
    
    num_videos = imdb.num_videos
    num_act_classes = imdb.num_actclasses
    
    sample=seqDataset[0]
    seqLength,featLength=sample['data'].shape[0:2]
    
    #Define the dataloader
    dataloader = DataLoader(seqDataset, batch_size=1,shuffle=False,num_workers = cfg.TEST.NUM_WORKERS)
    
    '''Read the model'''
    #Model parameters
    nMap=int(sample['smap'].shape[1])
    height,width=sample['smap'].shape[2:]
    #Create the model
    pmodel=sq.AMTGAF(featLength,cfg.ATT_SIZE,cfg.ZLENGTH,num_act_classes,cfg.GRID_SIZE,nMap=nMap,mapOp=cfg.MAP_OP,dropout=cfg.DROPOUT,temporal_horizon=cfg.TEMPORAL_HORIZON)
    pmodel.load_state_dict(torch.load(trained_model),strict=True)
    pmodel=pmodel.to(device)    
    pmodel.eval()
    
    """Losses"""
    GazeLoss=nn.MSELoss()
    
    """Performance metrics"""
    v_gLoss = np.zeros((num_videos,), dtype = np.float32)
    predV = np.zeros((num_videos,num_act_classes), dtype = np.float32)
    labelV = np.zeros((num_videos,), dtype = np.float32)

    
    successes=0
    cases=0
    
    gouts=[]
    glabels=[]
    gSuccess=np.zeros((num_videos,), dtype = np.float32)
    
    #If we do automatic per-user normalization instead of corpus-level normalization
    if cfg.AUTOMATIC_NORM:
        flimits=sample['flimits']
        numF=len(flimits)
        mom1=torch.zeros(numF,requires_grad=False);
        mom2=torch.zeros(numF,requires_grad=False)
        seqDataset.setNormalization(cfg.FMEANS,cfg.FSTDS)
        contIters=0    
        for sample in dataloader:
            mom1,mom2=updateMoms(sample['data'],mom1,mom2,flimits)
            contIters+=1    
        means, stds=computeNorms(mom1,mom2,contIters)
        seqDataset.setNormalization(means.cpu().numpy(),stds.cpu().numpy())    
    
    v=0;    
    for sample in dataloader:
        
        """Prepare data for the network"""
        smap=sample['smap'].to(device)
        labels=sample['labels'][0,...].to(device)
        data=sample['data'].to(device)        
        gt_fix=sample['gaze'].to(device)
        nFr=len(labels)
        
        '''Run the network'''
        outs = pmodel(data,smap)
        
        '''Get the outputs'''
        outputs=outs['act_preds']
        outputs=F.softmax(outputs,dim=2)
        
        '''Compute gaze loss'''
        fixations = outs['fixations']
        gLoss=2*GazeLoss(fixations,gt_fix)
        
            
        #Remove the fist dimension (video)       
        scores_action=outputs[0,...]   
      
        #Update        
        v_gLoss[v] = gLoss
        preds=torch.argmax(scores_action.view(-1,num_act_classes),dim=1)
        preds=preds[cfg.TEST.PRE_FRAMES:]
        labels_acc=labels.flatten()[cfg.TEST.PRE_FRAMES:]
        
        """Compute the predictions using the detection window"""
        opreds=preds.clone()
        contC=0
        for i in range(1,len(preds)):
            if preds[i]!=opreds[i-1]:
                contC+=1
            else:
                contC+=0
                
            if contC>=cfg.TEST.DETECTION_WINDOW:
                opreds[i]=preds[i]
                contC=0
            else:
                opreds[i]=opreds[i-1]
        preds=opreds
        
        
        labelV[v]=labels.max()
        """Compute global outputs"""
        gouts.append(preds)
        glabels.append(labels_acc)
        """Evaluate video performance"""
        vsuccesses=(preds==labels_acc).sum()
        vcases=labels_acc.numel()
        successes+=vsuccesses
        cases+=vcases
        
        
        if cfg.TEST.PRE_FRAMES>0:
            predV[v,...]=scores_action.view(-1,num_act_classes)[cfg.TEST.PRE_FRAMES:,...].sum(dim=0).cpu().numpy()
        else:
            predV[v,...]=scores_action.view(-1,num_act_classes).sum(dim=0).cpu().numpy()
            
        acierto=int(np.argmax(predV[v,...])==labelV[v])
        
        
        print('Video {:d}/{:d} with {:d} frames {:.3f}s gLoss {:.3f} accuracy {:d}'.format(v, num_videos, nFr, _t['fr_decision'].average_time, gLoss,acierto))
        
        #Success rate
        succv=(preds==labelV[v])
        idx_success=torch.where(succv)[0]
        if len(idx_success)>0:
            idx_success=idx_success[0]
            if idx_success>=0 and torch.all(succv[idx_success:]):
                gSuccess[v]=1;
            
      
        v+=1
    
    
    """Compute global performance metrics"""
    gouts=torch.cat(gouts,0)
    glabels=torch.cat(glabels,0)
    SR=gSuccess.sum()/len(gSuccess)
    fAcc=(gouts==glabels).sum()/len(gouts)
    print('average fAcc-SR: %.3f %.3f\n' %(fAcc,SR))
 


        
        
