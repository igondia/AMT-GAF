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
import cv2
import os

import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
from seq_data_layer.SeqDataset import SeqDataset
from torch.utils.data import DataLoader
from torch.nn import functional as F
import torch.nn as nn
import scipy.special
import sklearn.metrics as skm

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
    
    #Convert roidb to a ROIDataset that can be decoded by Pytorch
    seqDataset = SeqDataset(seqdb,imdb.actclasses,imdb.actclasses,imdb.objclasses,test=True)
    
    num_videos = imdb.num_videos
    num_obj_classes = imdb.num_objclasses
    num_act_classes = imdb.num_actclasses
    
    sample=seqDataset[0]
    nFeat=len(sample['data'])
    seqLength,featLength=sample['data'].shape[0:2]
    
    #Define the dataloader
    dataloader = DataLoader(seqDataset, batch_size=1,shuffle=False,num_workers = cfg.TEST.NUM_WORKERS)
    
   
    
    import seqmodels.amt_gaf as sq
    #Model parameters
    nMap=int(sample['smap'].shape[1])
    height,width=sample['smap'].shape[2:]
    #Create the model
    pmodel=sq.AMTGAF(featLength,cfg.ATT_SIZE,cfg.ZLENGTH,num_act_classes,cfg.GRID_SIZE,nMap=nMap,mapOp=cfg.MAP_OP,dropout=cfg.DROPOUT,temporal_horizon=cfg.TEMPORAL_HORIZON)
    # pmodel=sq.AMTGAF(active_feats,featLength,att_size,zlength,num_actclasses,cfg.GRID_SIZE,nMap=nMap,mapOp=cfg.MAP_OP,

    GazeLoss=nn.MSELoss()
    
    pmodel.load_state_dict(torch.load(trained_model),strict=True)
    pmodel=pmodel.to(device)    
    pmodel.eval()
    
    v_AUC = np.zeros((num_videos,), dtype = np.float32)
    v_gLoss = np.zeros((num_videos,), dtype = np.float32)
    predV = np.zeros((num_videos,num_act_classes), dtype = np.float32)
    labelV = np.zeros((num_videos,), dtype = np.float32)
    
    numCortes = len(cfg.TIME_MARKS)
    THs=np.arange(0,1.00,0.025)
    print(THs)
    numThs=THs.shape[0]
    
    
    
    counter = 0
    gLoss = 0
    auc = 0
    nMap=4
    nEMap=nMap+num_obj_classes-1
    tgaze_patterns = np.zeros((256, nMap), dtype = np.float32)
    cont_tgaze = np.zeros((256, nMap), dtype = np.float32)
    v=0;
    
    successes=0
    cases=0
    
    gouts=[]
    glabels=[]
    gSuccess=np.zeros((num_videos,), dtype = np.float32)
    
    if cfg.AUTOMATIC_NORM:
        mom1=torch.zeros(9,requires_grad=False);
        mom2=torch.zeros(9,requires_grad=False)
        fweights=np.array(cfg.INPUT_WEIGHTS.copy())
        fweights[...]=1.0
        #Setting 1s to fweights to compute right values
        seqDataset.setNormalization(cfg.FMEANS,cfg.FSTDS,fweights)
        contIters=0    
        for sample in dataloader:
            mom1,mom2=updateMoms(sample['data'][0],mom1,mom2,num_obj_classes)
            contIters+=1    
        means, stds=computeNorms(mom1,mom2,contIters)
        seqDataset.setNormalization(means.cpu().numpy(),stds.cpu().numpy(),np.array(cfg.INPUT_WEIGHTS))    
        
    for sample in dataloader:
        
        scores=sample['scores'].to(device)
        smap=sample['smap'].to(device)
        time=sample['time'][0,...].to(device)
        labels=sample['labels'][0,...].to(device)
        gLabels=sample['gaze'].to(device)
        # vfeat=sample['vfeat'].to(device)
        data=sample['data'].to(device)        
        
        
        nFr = data[0].shape[1]
        scores_action = np.zeros((nFr, num_act_classes), dtype = np.float32)
        gaze_patterns = np.zeros((nFr, nMap+1), dtype = np.float32)
        
        
        outs = pmodel(data,smap)
        outputs=outs['act_preds']
       
        outputs=F.softmax(outputs,dim=2)
        
        
       
        fixations = outs['fixations']
        gt_fix=sample['gaze'].to(device)
            
        # pdb.set_trace()
        gLoss=2*GazeLoss(fixations,gt_fix)
        auc=0
            

        counter+=1
            
        #Remove the fist dimension (video)       
        scores_action=outputs[0,...]   
        if cfg.TEST.SAVE_GAZE_PATTERNS:
            # pdb.set_trace()
            try:
                map_weights=outs['interpretable_w']
                aux_gaze_pattern = map_weights[0,...].cpu().numpy()
                gaze_patterns[..., 0:nMap] = aux_gaze_pattern[...,0:nMap]
                gaze_patterns[..., nMap] = aux_gaze_pattern[...,nMap:].max(axis=1)
                # gaze_patterns = gaze_patterns/gaze_patterns.sum(axis=1,keepdims=True)
                
                gaze_patterns =scipy.special.softmax(gaze_patterns,axis=1)
            except:
                gaze_patterns[...]=0
            
            fig1 = plt.figure(figsize=(14,8))
            ax = plt.subplot(111)
            timeax=np.linspace(0,gaze_patterns.shape[0]/25.0,gaze_patterns.shape[0]); 
            ax.plot(timeax,gaze_patterns,linewidth=3)
            ax.plot(timeax,labels.cpu().numpy()>0,linestyle='dashed',linewidth=3)
            box = ax.get_position()
            ax.set_position([box.x0, box.y0 + box.height * 0.1, box.width, box.height*0.9])
            # Put a legend to the right of the current axis
            ax.set_xlabel('Time (secs)')
            ax.set_ylabel('Weights')
            ax.legend(('Fixation', 'Center bias','Predictable Motion','Saccade','VWM','Grasping Action'),loc='upper center', bbox_to_anchor=(0.5, -0.08),fancybox=True, shadow=True, ncol=3)
            
            plt.title('Weights of eye motion patterns used to predict visual attention along time for video %d'%v)
            fig1.savefig('visualized_output/rgaze_patterns%d.png'%v)
            plt.close('all')

        #Update        
        v_AUC[v] = auc
        v_gLoss[v] = gLoss
        preds=torch.argmax(scores_action.view(-1,num_act_classes),dim=1)
        preds=preds[cfg.TEST.PRE_FRAMES:]
        labels_acc=labels.flatten()[cfg.TEST.PRE_FRAMES:]
        
        
        opreds=preds.clone()
        # print(preds)
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
        gouts.append(preds)
        glabels.append(labels_acc)
        
        vsuccesses=(preds==labels_acc).sum()
        vcases=labels_acc.numel()
        successes+=vsuccesses
        cases+=vcases
        
        acierto_fr=vsuccesses/vcases
        
        # if acierto_fr<0.5:
        #     # print(seqdb['video_data'][v]['data']['hand_score'][:,0].T)
        #     print(preds)
        #     print(labels.max())
            
        #     pdb.set_trace()
        
            
        # predV[v,...]=scores_action.view(-1,num_act_classes).sum(dim=0).cpu().numpy()
        if cfg.TEST.PRE_FRAMES>0:
            predV[v,...]=scores_action.view(-1,num_act_classes)[cfg.TEST.PRE_FRAMES:,...].sum(dim=0).cpu().numpy()
        else:
            predV[v,...]=scores_action.view(-1,num_act_classes).sum(dim=0).cpu().numpy()
            
        
        
        acierto=int(np.argmax(predV[v,...])==labelV[v])
        
        
        print('Video {:d}/{:d} with {:d} frames {:.3f}s gLoss {:.3f} acierto {:d} acierto_fr {:.2f}'.format(v, num_videos, nFr, _t['fr_decision'].average_time, gLoss,acierto,acierto_fr))
        video_label = int(labels.cpu().numpy().max())
        # (atp, afp, apos, apred) = computeDetections(scores_action.cpu().numpy(), scores_obj.cpu().numpy(), labels.cpu().numpy(), time.cpu().numpy(),cfg.TEST.DETECTION_TH)
        # saveOutputVideo(v,seqdb,imdb,apred,gt_fix.cpu().numpy()[0,...],fixations.cpu().numpy()[0,...],atp.sum(),afp.sum(),video_label)
        
        #Success rate
        succv=(preds==labelV[v])
        idx_success=torch.where(succv)[0]
        if len(idx_success)>0:
            idx_success=idx_success[0]
            if idx_success>=0 and torch.all(succv[idx_success:]):
                gSuccess[v]=1;
            
        # print(preds)
        
        # print(labels.max())
        # print((float(acierto_fr),gSuccess[v]))
        # pdb.set_trace()
        v+=1
    
    
    # evaluateResults(predV,labelV,imdb.actclasses)        
    # fAcc=successes/cases
    # vAcc=(np.argmax(predV,axis=1)==labelV).mean()
    gouts=torch.cat(gouts,0)
    glabels=torch.cat(glabels,0)
    SR=gSuccess.sum()/len(gSuccess)
    #Global metrics
    fAcc=(gouts==glabels).sum()/len(gouts)
    # pdb.set_trace()
    # f1 = skm.f1_score(glabels.cpu().numpy(), gouts.cpu().numpy())
    print('average fAcc-SR: %.3f %.3f\n' %(fAcc,SR))
 

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
    tp = np.zeros((num_obj_classes, 1), dtype = np.float32)
    fp = np.zeros((num_obj_classes, 1), dtype = np.float32)
    pos = np.zeros((num_obj_classes, 1), dtype = np.float32)
    action_pred = np.zeros((ypred.shape[0],), dtype = np.float32)
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
#                    #Multiclass
                    #Case 1: The detection is correct in time and label
                    if detected_class==video_label and tp[detected_class]==0:
                        tp[detected_class]=np.maximum(time[f],1e-10);
                    #Case 2: We have a wrong detection at any time
                    elif detected_class!=video_label:
                        fp[detected_class]+=1;
                    #Binario
#                    if tp[detected_class]==0:
#                        tp[video_label]=time[f];
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


def ComputePrecRecall(tp, fp, pos):
    numCortes = len(cfg.TIME_MARKS)
    prec = np.zeros((numCortes,), dtype = np.float32)
    recall = np.zeros((numCortes,), dtype = np.float32)
   
    for c in range(numCortes):
        aux_tp = ((tp < cfg.TIME_MARKS[c]) & (tp > 0)).astype(int).sum()
        prec[c] = aux_tp / (aux_tp + fp.sum() + 1e-05)
        recall[c] = aux_tp / (pos.sum() + 1e-05)
    
    return (prec, recall)

def ComputeEvalMetrics(rdata):
    
    numCortes = len(cfg.TIME_MARKS)
    numThs = len(rdata)
    
    AP = np.zeros((numCortes,), dtype = np.float32)
    Fscore = np.zeros((numCortes,), dtype = np.float32)
    prec = np.zeros((numCortes,), dtype = np.float32)
    recall = np.zeros((numCortes,), dtype = np.float32)
    
    for c in range(numCortes):
        trecall=np.zeros((numThs,), dtype = np.float32)
        tprec=np.zeros((numThs,), dtype = np.float32)
        for t in range(numThs):
            tp=rdata[t]['tp']
            fp=rdata[t]['fp']
            pos=rdata[t]['pos']    
            aux_tp = ((tp < cfg.TIME_MARKS[c]) & (tp > 0)).astype(int).sum()
            tprec[t] = aux_tp / (aux_tp + fp.sum() + 1e-05)
            trecall[t] = aux_tp / (pos.sum() + 1e-05)
            if tprec[t]==0 and trecall[t]==0:
                tprec[t]=1
     
        
        Fscore[c]=0
        #Average Precision
        trecall=trecall[::-1]
        tprec=tprec[::-1]
        prev_recall=0;
        prev_t=0
        for t in range(numThs):
            if trecall[t]>prev_recall:
                aux_tprec=tprec[prev_t:t+1].mean() #the average of the intermediate precision
                AP[c]+=(trecall[t]-prev_recall)*aux_tprec
                prev_recall=trecall[t]
                prev_t=t
                F = 2.0 * aux_tprec * trecall[t] / (aux_tprec + trecall[t] + 1e-100)
                if F > Fscore[c]:
                    Fscore[c]=F
                    prec[c]=aux_tprec
                    recall[c]=trecall[t]
                    best_th=t
        
        # plt.plot(tprec,trecall)
        # plt.axis([0, 1, 0, 1])
        # plt.grid(True)
        # plt.xlabel('precision')
        # plt.ylabel('recall')
        # plt.show()
        
    return recall,prec,Fscore,AP

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


def evaluateResults(scores,labels,classes_names):
    num_classes=scores.shape[1]
    #Computing scores per_category
    aps=np.zeros(num_classes,)
    ypred=scores.argmax(axis=1)
    cm=skm.confusion_matrix(labels,ypred,labels=range(num_classes)).astype(float)
    ncm=np.transpose( np.transpose(cm) / np.maximum(cm.astype(float).sum(axis=1),1e-40) )
    print('Final results:')
    print('*****************')    
    for cat in range(num_classes):
        scores_cat=scores[...,cat]
        labels_cat=(labels==cat).astype(float)
        if(labels_cat.sum()>0):
            aps[cat]=skm.average_precision_score(labels_cat,scores_cat)
        else:
            aps[cat]=0
        print('%25s:\tap: %.3f\tacc: %.3f'%(classes_names[cat],aps[cat],ncm[cat,cat]))
    print('*****************')
    diagonal=np.diag(ncm)
    print('%25s:\tap: %.3f-%.3f\tacc: %.3f\tbacc: %.3f-%.3f'%('Average',aps.mean(),aps.std(),cm.trace()/cm.sum(),diagonal[cm.sum(axis=1)>0].mean(),diagonal[cm.sum(axis=1)>0].std()))    
    print('Confusion Matrix')
    np.set_printoptions(precision=0,linewidth=400,)
    short_names=[x[0:4] for x in classes_names[1:]]
    short_names=' ' + ' '.join(short_names)
    print('{0:<15} {1}'.format('Categories',short_names))
    
    
    for cat in range(num_classes):
        print('{0:<12} {1:}'.format(classes_names[cat],' '.join(np.char.mod('%3d', cm[cat,:].astype(int))))) 
    print('Normalized Confusion Matrix')  
    np.set_printoptions(precision=0,linewidth=400,)
    print('{0:<15} {1}'.format('Categories',short_names))
    for cat in range(num_classes):
        print('{0:<12} {1:}'.format(classes_names[cat],' '.join(np.char.mod('%3d', np.round(100*ncm[cat,:]).astype(int))))) 
        
        
def updateMoms(data,mom1,mom2,num_actclasses):
    idxc=0
    idx=range(idxc,idxc+2);
    
    #Fix position
    mom1[0]+=data[...,idx].mean()
    mom2[0]+=(data[...,idx]**2).mean()
    idxc+=2
    
    #Fix motion
    idx=range(idxc,idxc+2);
    mom1[1]+=data[...,idx].mean()
    mom2[1]+=(data[...,idx]**2).mean()
    
    #IMU
    idxc=idxc+2;
    imu_off=6 
    #sin imu imu_off=2
    # imu_off=2
    idx=range(idxc,idxc+imu_off);
    mom1[2]+=data[...,idx].mean()
    mom2[2]+=(data[...,idx]**2).mean()
    idxc+=imu_off
    
    #previous map scores
    idx=range(idxc,idxc+5);
    mom1[3]+=data[...,idx].mean()
    mom2[3]+=(data[...,idx]**2).mean()
    idxc+=5
    
    # ao scores
    idx=range(idxc,idxc+num_actclasses);
    mom1[4]+=data[...,idx].mean()
    mom2[4]+=(data[...,idx]**2).mean()
    idxc+=num_actclasses;
    
    #VWM scores
    idx=range(idxc,idxc+2*(num_actclasses-1),2);
    mom1[5]+=data[...,idx].mean()
    mom2[5]+=(data[...,idx]**2).mean()
    
    #VWM distances
    idx=range(idxc+1,idxc+2*(num_actclasses-1),2);
    mom1[6]+=data[...,idx].mean()
    mom2[6]+=(data[...,idx]**2).mean()
    idxc=idxc+2*(num_actclasses-1)
    
    # #Hand score
    # idx=range(idxc,idxc+1);
    # mom1[7]+=data[...,idx].mean()
    # mom2[7]+=(data[...,idx]**2).mean()
    # idxc=idxc+1
    
    # #Hand distance
    # idx=range(idxc,idxc+1);
    # mom1[8]+=data[...,idx].mean()
    # mom2[8]+=(data[...,idx]**2).mean()
    # idxc=idxc+1
    
    #Action score
    idx=range(idxc,idxc+1);
    mom1[7]+=data[...,idx].mean()
    mom2[7]+=(data[...,idx]**2).mean()
    idxc=idxc+1
    
    #Gaze std
    idx=range(idxc,idxc+2);
    mom1[8]+=data[...,idx].mean()
    mom2[8]+=(data[...,idx]**2).mean()
    
    return mom1,mom2

def computeNorms(mom1,mom2,contIters):
    mom1=mom1/contIters
    mom2=mom2/contIters
    means=mom1;
    stds=np.sqrt(mom2-mom1**2)
    print(means)
    print(stds)
    return means,stds