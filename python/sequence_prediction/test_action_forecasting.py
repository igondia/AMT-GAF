# --------------------------------------------------------
# AMT-GAF
# Licensed under BSD-2 License
# Written by Iván González
# --------------------------------------------------------

'''Test a Fast R-CNN network on an imdb (image database).'''
import pdb
import warnings
warnings.filterwarnings("ignore")

from sequence_prediction.config import cfg, get_output_dir
import seq_data_layer.seqdb as sdl_seqdb
from utils.timer import Timer
import numpy as np
import cv2
import os
import seqmodels.amt_gaf as sq
import matplotlib.pyplot as plt
import torch
from seq_data_layer.SeqDataset import SeqDataset
from torch.utils.data import DataLoader
from torch.nn import functional as F
import torch.nn as nn
from performance_metrics.metrics import ComputeEvalMetrics

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
    
    
    """GEnerate the dataset"""
    sdl_seqdb.prepare_seqdb(imdb)
    seqdb = imdb.seqdb
    seqDataset = SeqDataset(seqdb,imdb.actclasses,imdb.actclasses,imdb.objclasses,test=True)
    
    num_videos = imdb.num_videos
    num_obj_classes = imdb.num_objclasses
    num_act_classes = imdb.num_actclasses
    
    sample=seqDataset[0]
    seqLength,featLength=sample['data'].shape[0:2]
    
    #Define the dataloader
    dataloader = DataLoader(seqDataset, batch_size=1,shuffle=False,num_workers = cfg.TEST.NUM_WORKERS)
    
    """Create the model"""
    att_size=cfg.ATT_SIZE
    zlength=cfg.ZLENGTH
    nMap=int(sample['smap'].shape[1])
    height,width=sample['smap'].shape[2:]
    pmodel=sq.AMTGAF(featLength,att_size,zlength,num_act_classes,cfg.GRID_SIZE,nMap=nMap,mapOp=cfg.MAP_OP,temporal_horizon=cfg.TEMPORAL_HORIZON)
    
    """Load the model"""     
    pmodel.load_state_dict(torch.load(trained_model),strict=True)
    pmodel=pmodel.to(device)    
    pmodel.eval()
    
    """Losses and performance metrics"""
    GazeLoss=nn.MSELoss()
    v_AUC = np.zeros((num_videos,), dtype = float)
    v_gLoss = np.zeros((num_videos,), dtype = float)
    numCortes = len(cfg.TIME_MARKS)
    THs=np.arange(0,1.00,0.025)
    numThs=THs.shape[0]
    rdata=[]
    for t in range(numThs):
        tp = np.zeros((num_obj_classes, 0), dtype = float)
        fp = np.zeros((num_obj_classes, 1), dtype = float)
        pos = np.zeros((num_obj_classes, 1), dtype = float)
        rdata.append({'tp': tp, 'fp': fp,'pos': pos})
       
    gLoss = 0
    auc = 0
    nMap=4
    v=0
    for sample in dataloader:
        
        scores=sample['scores'].to(device)
        smap=sample['smap'].to(device)
        time=sample['time'][0,...].to(device)
        labels=sample['labels'][0,...].to(device)
        data=sample['data'].to(device)       
        gt_fix=sample['gaze'].to(device)
        
        
        nFr = data.shape[1]
        
        """Run the model"""
        outs = pmodel(data,smap)
        
        """Get action prediction probs"""
        outputs=outs['act_preds']
        outputs=F.softmax(outputs,dim=2)
        scores_action=outputs[0,...]  
        
        """Compute gaze Loss"""
        fixations = outs['fixations']
        gLoss=2*GazeLoss(fixations,gt_fix)
        v_gLoss[v] = gLoss
        
        """Object scores"""
        scores_obj = scores[0, ...]
        
        """Compute Detections at different threholds"""
        for t,th in enumerate(THs):
            (atp, afp, apos, apred) = computeDetections(scores_action.cpu().numpy(), scores_obj.cpu().numpy(), labels.cpu().numpy(), time.cpu().numpy(),th)
            rdata[t]['tp'] = np.hstack((rdata[t]['tp'], atp))
            rdata[t]['fp'] += afp
            rdata[t]['pos'] += apos
        
        print('Video {:d}/{:d} with {:d} frames {:.3f}s gLoss {:.3f}'.format(v, num_videos, nFr, _t['fr_decision'].average_time, gLoss))
        
        # For visualization
        # video_label = int(labels.cpu().numpy().max())
        # (atp, afp, apos, apred) = computeDetections(scores_action.cpu().numpy(), scores_obj.cpu().numpy(), labels.cpu().numpy(), time.cpu().numpy(),cfg.TEST.DETECTION_TH)
        # saveOutputVideo(v,seqdb,imdb,apred,gt_fix.cpu().numpy()[0,...],fixations.cpu().numpy()[0,...],atp.sum(),afp.sum(),video_label)
        
        v+=1
        
    
    recall,prec,Fscore,AP = ComputeEvalMetrics(rdata,cfg.TIME_MARKS)   
    
            
    #Last value
    print('precs:     \t' + np.array2string(prec[...], formatter = {
        'float_kind': (lambda x: '%.2f' % x) }))
    print('recalls:   \t' + np.array2string(recall[...], formatter = {
        'float_kind': (lambda x: '%.2f' % x) }) )
    print('F-measures:\t' + np.array2string(Fscore[...], formatter = {
        'float_kind': (lambda x: '%.2f' % x) }) )
    print('AP-measures:\t' + np.array2string(AP[...], formatter = {
        'float_kind': (lambda x: '%.2f' % x) }) )

    print('average prec-rec-F-AP-GazeLoss: %.2f %.2f %.2f %.3f %.3f\n' %(prec.mean(),recall.mean(),Fscore.mean(),AP.mean(),v_gLoss.mean()))
 

def computeDetections(scores_action, scores_obj, labels_object, time,TH):
    num_obj_classes = scores_obj.shape[1]
    num_act_classes = scores_action.shape[1]
    #remove the bg score
    scores_action [:,0] = 0
    video_label = int(labels_object.max())
    if num_act_classes == 2:
        ypred = scores_obj.argmax(axis = 1) * scores_action.argmax(axis = 1)
    else:
        ypred = scores_action.argmax(axis = 1)
        outs = scores_action.max(axis = 1)
        
    actionFr = np.nonzero(labels_object)[0][0]
    nFr = labels_object.shape[0]
    detected_class = -1
    count_det = 0
    tp = np.zeros((num_obj_classes, 1), dtype = float)
    fp = np.zeros((num_obj_classes, 1), dtype = float)
    pos = np.zeros((num_obj_classes, 1), dtype = float)
    action_pred = np.zeros((ypred.shape[0],), dtype = float)
    f = 0
    
    if np.any(labels_object<0):
         return (tp, fp, pos, action_pred)
     
    while 1:
         #If it is a grasping action and the score is above the threshold
         if ypred[f]>0 and outs[f]>TH:
            #Check if we are detecting or not
            if ypred[f]==detected_class:
                count_det+=1;
            else:
                count_det=0;
                detected_class=ypred[f];

            # We have a detection, we need to check if it is correct or not    
            if count_det>=cfg.TEST.DETECTION_WINDOW:
                
                action_pred[f]=detected_class
                if f>=(actionFr-cfg.TEST.PRE_FRAMES):
                   #Multiclass
                    #Case 1: The detection is correct in time and label
                    if detected_class==video_label and tp[detected_class]==0:
                        tp[detected_class]=np.maximum(time[f],1e-10);
                    #Case 2: We have a wrong detection at any time
                    elif detected_class!=video_label:
                        fp[detected_class]+=1;
                   
                #Case 3: We have a wrong detection if the labels is true quite before the right time     
                else:
                    fp[detected_class]+=1;
                #We reset the count_det and the detection class
                count_det=0;
                
                 #We look for the last frame with this object
                newf=np.nonzero(ypred[f+1:]!=detected_class)[0];
                if newf.shape[0]>0:
                    newf=f+newf[0]+1;
                    action_pred[f:newf]=action_pred[f]
                    f=int(np.maximum(newf,f+cfg.TEST.DETECTION_GUARD))
                else:
                    action_pred[f:nFr]=action_pred[f]
                    f=nFr;
                detected_class=-1;    
         f+=1;
         if(f>=nFr):
            break
    
    pos[video_label] = 1
    
    return (tp, np.minimum(fp, 1), np.minimum(pos, 1), action_pred)




def saveOutputVideo(v,seqdb,imdb,apred,fix_gt,fix_pred,tp,fp,video_label):
    nFr = apred.shape[0]
    
    '''OUTPUT VIDEO'''
    if tp>0 and fp==0:
        videoName='outputs_video/success%d.avi'%v
    else:
        videoName='outputs_video/error%d.avi'%v
    frFile=seqdb['video_data'][v]['images'][0]
    im = cv2.imread(frFile)
        
    width=im.shape[1]
    height=im.shape[0]
    fourcc = cv2.VideoWriter_fourcc(*'X264')
    videoStream = cv2.VideoWriter(videoName,fourcc,15,(width,height))
    text=''
    prev_class_det=-1
    class_det=-1
    for f in range(nFr):
         #We read the frame
        frFile=seqdb['video_data'][v]['images'][f]
        #        print frFile[-70:]
        im = cv2.imread(frFile)
        (h,w,channels)=im.shape
        curr_fix=seqdb['video_data'][v]['data']['fix'][f].astype(int);
        im=cv2.circle(im, (curr_fix[0],curr_fix[1]), 5, (255,0,0), 10) 
        gt_fix=fix_gt[f,:]
        gt_fix=((gt_fix+0.5)*np.array((w,h),dtype=np.float32)).astype(int)
        im=cv2.circle(im, (gt_fix[0],gt_fix[1]), 5, (0,255,0), 10) 
        fix=fix_pred[f,:]
        fix=((fix+0.5)*np.array((w,h),dtype=np.float32)).astype(int)
        im=cv2.circle(im, (fix[0],fix[1]), 5, (0,0,255), 10) 
        if f==0:                    
            im=cv2.putText(im, 'Gaze', (fix[0]+50,fix[1]), cv2.FONT_HERSHEY_SIMPLEX, 2, (255,0,0),5)            
            for aux in range(15):
                videoStream.write(im)

        if(apred[f]>0): #and text=='':
            class_det=int(apred[f])
            text='Grasp ' + imdb.actclasses[class_det]
        if len(text)<15:
            Fsize=3
        else:
            Fsize=2
            
        if(class_det==video_label):
            tColor=(0,255,0)
        else:
            tColor=(0,0,255)
            
        im=cv2.putText(im, text, (100,100), cv2.FONT_HERSHEY_SIMPLEX, Fsize, tColor,5)
        if(class_det!=prev_class_det):
            prev_class_det=class_det
            for aux in range(15):
                videoStream.write(im)
        
        videoStream.write(im)
    #We add last frames
    for ef in range(1000):
        frFile=imdb.image_path_at(v,nFr+ef,0)            
        if os.path.exists(frFile)==0:
            break
        im = cv2.imread(frFile)
        im=cv2.putText(im, text, (100,100), cv2.FONT_HERSHEY_SIMPLEX, Fsize, tColor,5)
        videoStream.write(im)
    videoStream.release()        
