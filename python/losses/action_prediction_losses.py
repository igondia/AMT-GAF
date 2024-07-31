# --------------------------------------------------------
# AMT-GAF
# Licensed under BSD-2 License
# Written by Iván González
# --------------------------------------------------------
import pdb
import torch
from torch import Tensor
 
    
"""Weak Action Loss => It accounts for two challenges in our weakly annotated scenario:
    1) the temporal annotations are only approximately aligned with the grasping intention
    2) active object scores (input features), as they come from other automatic module, might be wrong and lead to overfitting"""
class WeakSeqLossLayer():
    

    def __init__(self,num_classes,preFrames=0, minFramesDet=3, mul=0):
        
        self.num_classes=num_classes
        self.preFrames=int(preFrames)
        self.minFramesDet=int(minFramesDet)
        self.eps=1e-20;
        self.BGW=1.0;#/self.num_classes;
        self.toffset=0
        self.mul = mul;        
    def __call__(self, action_score: Tensor, target: Tensor, time: Tensor, scores: Tensor) -> Tensor:
        device=action_score.device
        
        (nV,nFr,numCat)=action_score.shape
        Loss=torch.tensor(0.0,device=device,requires_grad=True)
        for v in range(nV):
            
            vtarget=target[v,...]
            vaction_score=action_score[v,...]
            vtime=time[v,...]
            vscores=scores[v,...]
            
            
            #Active object in the sequence
            active_object=int(vtarget.max())
            
            #If there is no target we do not learn
            if active_object<=0:
                continue;
                
            #Sequence start and end
            seqStart=torch.nonzero(vtime>=0)[0][0]
            seqEnd=torch.nonzero(vtime>=0)[-1][0]+1
            #real NFR to be analyzed
            realNFR=seqEnd-seqStart;
            
            #Frames where the active object is fixated
            fixIds=torch.nonzero(vtarget)[0]
            if len(fixIds)>0:
                fixStart=fixIds[0]
                seqEnd=torch.max(seqEnd,fixStart+1);
            else:
                fixStart=-1;
            
            #Now crop sequences
            vscores=vscores[seqStart:seqEnd,:]
            vaction_score=vaction_score[seqStart:seqEnd,:]
            vtarget=vtarget[seqStart:seqEnd]
            nFr=seqEnd-seqStart
            
            del seqStart
            del seqEnd
            
            #objects score is set to 1
            pos_weight=torch.ones((nFr,self.num_classes),device=device,requires_grad=False)
            #Assigning time matrix
            neg_weight=torch.ones((nFr,self.num_classes),device=device,requires_grad=False)
            #Weights considering a fixated object
            if fixStart>=0: 
                x=vscores[fixStart:,...].clone()
                x[...,0]=0
                ao_per_frame=torch.argmax(x,dim=1);
                framesAO=(ao_per_frame==active_object);
                numFixationsAO=torch.cumsum(framesAO,axis=0)
                timesAO=numFixationsAO.max()
                #The active object should appear at least minFramesDet frames
                # Wrong sequence => No learn
                if(timesAO<self.minFramesDet):
                    #We do not learn anythong from the fixStart til the end of the sequence
                    limitFr=torch.max(nFr-self.preFrames,fixStart)
                #Right sequence => Learn
                else:
                    #Compute the real fixStart
                    limitFr=fixStart+torch.nonzero(numFixationsAO>=self.minFramesDet)[0][0]
                    minLearningFr=torch.max(nFr-self.preFrames,fixStart)
                    limitFr=torch.min(limitFr,minLearningFr)

                #We remove the influence of the preFrames on the loss => Only to GT object and Background
                pos_weight[fixStart:limitFr,active_object]=0
                neg_weight[fixStart:limitFr,0]=0
                
                # pdb.set_trace()    
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
                pos_weight[:fixStart,:]=pos_weight[:fixStart,:]*mul
                
             
           
            
            #Generate the labels matrix        
            labels=torch.zeros((nFr,self.num_classes),device=device,requires_grad=False)
            labels[range(nFr),vtarget]=1
            #normalize the weights
            total_weights=((labels*pos_weight)+(1.0-labels)*neg_weight).sum()
            normalizer=(nFr*numCat)/total_weights;
            neg_weight=normalizer*neg_weight
            pos_weight=normalizer*pos_weight

            #Sigmoid
            yp=torch.sigmoid(vaction_score)
            #Compute loss
            Loss=Loss-(labels*pos_weight*torch.log(yp+self.eps)+(1.0-labels)*neg_weight*torch.log(1.0-yp+self.eps)).mean();
        #Video-level Average loss   
        return Loss/nV                   
      
        
