# --------------------------------------------------------
# AMT-GAF
# Licensed under BSD-2 License
# Written by Iván González
# --------------------------------------------------------

import numpy as np
import pdb
import torch
from torch import Tensor

"""Class the computes the accuracy of the action forecasting, considering some temporal misalignment between labels and video contents."""
class SeqAcc():
    

    def __init__(self,num_obj_classes=1,num_act_classes=2,minFramesDet=5,framesBetween=5,preFrames=5):
        self.num_obj_classes=num_obj_classes;
        self.num_act_classes=num_act_classes
        self.minFramesDet=minFramesDet
        self.framesBetween=framesBetween
        self.preFrames=preFrames
        

    def __call__(self, action_scores: Tensor, labels: Tensor, times: Tensor,scores: Tensor):
                
        with torch.no_grad():
            numVideos=action_scores.shape[0]
            tp=torch.zeros((self.num_obj_classes,1),device=action_scores.device,requires_grad=False)
            fp=torch.zeros((self.num_obj_classes,1),device=action_scores.device,requires_grad=False)
            pos=torch.zeros((self.num_obj_classes,1),device=action_scores.device,requires_grad=False)
            eps=torch.tensor(1e-10,device=action_scores.device,requires_grad=False)
                
        
            for v in range(numVideos):
                time=times[v,...]
                action_score=action_scores[v,time>=0]
                label=labels[v,time>=0]
                score=scores[v,time>=0]
                video_label=int(labels[v,time>=0].max())
                #In case we have removed the video
                if torch.any(label<0):
                    # print('Discarding  video')
                    # pdb.set_trace()
                    continue;
                    
                if self.num_act_classes==2:
                    score=action_score[:,2:]
                    ypred=score[0,:].argmax(axis=1)*action_score.argmax(axis=1)
                else:
                    ypred=action_score.argmax(axis=1)
               
                #First identify The starting point of the grasping action
                actionFrs=torch.nonzero(label)[:,0]
                if len(actionFrs)>0:
                    actionFr=actionFrs[0]
                else:
                    actionFr=np.inf;
                last_Frame=np.nonzero(time>=0)[-1][0]
                
                nFr=label.shape[0]
                #Second, look for detections
                detected_class=-1;
                count_det=0;
                f=0;
                
                while 1:
        
                    if ypred[f]>0:
                        #Check if we are detecting or not
                        if ypred[f]==detected_class:
                            count_det+=1;
                        else:
                            count_det=0;
                            detected_class=ypred[f];
                        # We have a detection, we need to check if it is correct or not    
                        if count_det>=self.minFramesDet:
                            if f>=(actionFr-self.preFrames):
                      	      #Case 1: The detection is correct in time and label
                      	      if detected_class==video_label and tp[detected_class]==0:
                      	          tp[detected_class]=torch.max(time[f],eps);
                      	      #Case 2: We have a wrong detection at any time
                      	      elif detected_class!=video_label:
                      	          fp[detected_class]+=1;
        #                   
                      		  #Case 3: We have a wrong detection if the labels is true quite before the right time     
                            else:
                            	fp[detected_class]+=1;
                             #We look for the last frame with this object
                            newf=np.nonzero(ypred[f+1:]!=detected_class);
                            if newf.shape[0]>0:
                                newf=f+newf[0][0].cpu().numpy()+1;
                                f=int(np.maximum(newf,f+self.framesBetween))
                            else:
                                f=nFr;
                            #We reset the count_det and the detection class
                            count_det=0;
                            detected_class=-1;
                                                       
                    f+=1;
                    if(f>=last_Frame):
                        break
                pos[video_label]+=1;
        fp[fp>1]=1.0
        pos[pos>1]=1.0
        return tp,fp,pos
    
"""Compute precision, recall for different values of anticipation time"""    
def ComputePrecRecall(tp,fp,pos,time_marks):
    numMarks=len(time_marks);
    prec=np.zeros((numMarks,),dtype=float)
    recall=np.zeros((numMarks,),dtype=float)
    for c in range(numMarks):
        aux_tp=((tp<time_marks[c])&(tp>0)).astype(dtype=int).sum();
        prec[c]=aux_tp/(aux_tp+fp.sum()+1e-5)
        recall[c]=aux_tp/(pos.sum()+1e-5)

    return (prec,recall)

"""Compute precision, recall, F-score and AP for different values of the anticipation time
precision, recall and F-score are obtained for the best detection threshold"""
def ComputeEvalMetrics(rdata,time_marks):
    
    numMarks = len(time_marks)
    numThs = len(rdata)
    
    AP = np.zeros((numMarks,), dtype = float)
    Fscore = np.zeros((numMarks,), dtype = float)
    prec = np.zeros((numMarks,), dtype = float)
    recall = np.zeros((numMarks,), dtype = float)
    
    for c in range(numMarks):
        trecall=np.zeros((numThs,), dtype = float)
        tprec=np.zeros((numThs,), dtype = float)
        for t in range(numThs):
            tp=rdata[t]['tp']
            fp=rdata[t]['fp']
            pos=rdata[t]['pos']    
            aux_tp = ((tp < time_marks[c]) & (tp > 0)).astype(int).sum()
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
        
    return recall,prec,Fscore,AP