# --------------------------------------------------------
# AMT-GAF
# Licensed under BSD-2 License
# Written by Iván González
# --------------------------------------------------------
import pdb
import numpy as np
import torch
from torch import Tensor

class SeqLossLayer():
    

    def __init__(self,num_classes,preFrames=0, mul=0):
        
        self.num_classes=num_classes
        self.preFrames=int(preFrames)
        self.eps=1e-20;
        self.BGW=1.0;#/self.num_classes;
        self.toffset=0
        self.mul = mul;        
    def __call__(self, action_score: Tensor, target: Tensor) -> Tensor:
        device=action_score.device
        
        (nV,nFr,numCat)=action_score.shape
        Loss=torch.tensor(0.0,device=device,requires_grad=True)
        for v in range(nV):
            vtarget=target[v,...]
            vaction_score=action_score[v,...]
            
            #Active object in the sequence
            target_act=int(vtarget.max())
            
            #If there is no target we do not learn
            if target_act<=0:
                continue;
                
            #Sequence start and end
            seqStart=0
            seqEnd=int(torch.nonzero(vtarget>=0)[-1][0]+1)
            
            #Frames where the active object is fixated
            fixStart=np.maximum(seqEnd-self.preFrames,0)
            
            #Now crop sequences
            vaction_score=vaction_score[seqStart:seqEnd,:]
            vtarget=vtarget[seqStart:seqEnd]
            nFr=seqEnd-seqStart
            
            #objects score is set to 1
            pos_weight=torch.ones((nFr,self.num_classes),device=device,requires_grad=False)
            #Assigning time matrix
            neg_weight=torch.ones((nFr,self.num_classes),device=device,requires_grad=False)
            # if target_act>0 and target_act%2==0:
            #     #During analysis, we accept either grasping object or viewing object, but not the rest
            #     pos_weight[seqStart:fixStart,target_act]=0
            #     neg_weight[seqStart:fixStart,target_act-1]=0
            
            pos_weight[seqStart:fixStart,:]=0
            neg_weight[seqStart:fixStart,:]=0
            
            if self.mul==0:
               #Compute proportions    
               frPos=nFr-fixStart#limitFr
               frNeg=fixStart
               mul = frPos/frNeg  
            else:
               mul = torch.tensor(self.mul)
               mul=torch.clip(mul,0.25,40)
                
               #Ajustamos proporciones
               neg_weight[:fixStart,:]=neg_weight[:fixStart,:]*mul
               pos_weight[fixStart:,:]=pos_weight[fixStart:,:]*mul
                
            
            #Generate the labels matrix        
            labels=torch.zeros((nFr,self.num_classes),device=device,requires_grad=False)
            labels[range(nFr),vtarget]=1
            #normalize the weights
            total_weights=((labels*pos_weight)+(1.0-labels)*neg_weight).sum()
            # normalizer=2.0*(nFr*numCat)/total_weights;
            normalizer=2.0/total_weights;
            neg_weight=normalizer*neg_weight
            pos_weight=normalizer*pos_weight
                        
            #Sigmoid
            yp=torch.sigmoid(vaction_score)
            #Compute loss
            Loss=Loss-(labels*pos_weight*torch.log(yp+self.eps)+(1.0-labels)*neg_weight*torch.log(1.0-yp+self.eps)).sum();
            # Loss_f=-(labels*pos_weight*torch.log(yp+self.eps)+(1.0-labels)*neg_weight*torch.log(1.0-yp+self.eps)).sum(dim=1)
            # print(vtarget)
            # print(torch.argmax(yp,dim=1))
            # print(Loss_f)
            # print((labels*pos_weight).sum())
            # pdb.set_trace()
        return Loss        
      
#Loss layer for EGTEA+ dataset, where non continous detection is performed      
class SeqEGTEALossLayer():
    

    def __init__(self,num_classes,preFrames=0, minFramesDet=1):
        
        self.num_classes=num_classes
        self.preFrames=int(preFrames)
        self.minFramesDet=int(minFramesDet)
        self.eps=1e-20;
        self.BGW=1.0;#/self.num_classes;
        self.toffset=0
                
    def __call__(self, action_score: Tensor, target: Tensor, time: Tensor, scores: Tensor) -> Tensor:
        device=action_score.device
        
        (nV,nFr,numCat)=action_score.shape
        Loss=0
        
        for v in range(nV):
            
            vtarget=target[v,...]
            vaction_score=action_score[v,...]
            vtime=time[v,...]
            # vtime=vtime/vtime.sum()
            vscores=scores[v,...]
            
            #Active object in the sequence
            active_object=int(vtarget.max())
            
            yp=torch.sigmoid(vaction_score)
            
            labels=torch.zeros((nFr,self.num_classes),device=device,requires_grad=False)
            labels[...,active_object]=1
            #We evaluate the last 
            Loss-=(vtime*(labels*torch.log(yp+self.eps)+(1.0-labels)*torch.log(1.0-yp+self.eps)).mean(dim=1)).sum()
            if torch.isnan(Loss):
                pdb.set_trace()
                
        return Loss        
