# -*- coding: utf-8 -*-

# -*- coding: utf-8 -*-

# --------------------------------------------------------
# CCNN 
# Copyright (c) 2018 [See LICENSE file for details]
# Written by Iván González
# --------------------------------------------------------
import pdb
#import ccnn
from sequence_prediction.config import cfg
import numpy as np
import sys
import torch
from torch import Tensor
from torch.nn import functional as F

        
class AsymmetricLoss():
    

    def __init__(self,weights, types, lossTH=1.0, eta_param=1e-2,maxIters=3):
        self.numLosses=len(weights)
        self.weights=weights
        self.types=types
        self.maxIters=maxIters
        self.lossTh=lossTH;
        self.eta_param=eta_param;
        
        self.iter_type=0;
        self.iter_counter=0;
        self.lambda_param=0.0;
        self.prev_lambda_param=self.lambda_param;
        self.t_eta_param=self.eta_param;
        self.prevLoss=torch.tensor(np.inf,device=weights.device,requires_grad=False)
        self.initLoss=0;
        self.numTypes=torch.max(self.types)+1;
        
        #Choose the first non-zero type
        self.iter_type=0
        self.t_eta_param=self.eta_param;
        
    def __call__(self, losses: Tensor, initialize=False) -> Tensor:
        
        with torch.no_grad():        
            if initialize:
                self.iter_type=0;
                
            #Inputs                
            tempWeights=self.weights.clone().detach()
             #Set tempWeights to the current iters
            tempWeights[self.iter_type!=self.types]=0;
            updateWeights=tempWeights>0
            #If we are in the regularizing loss (GAze Loss in our case)
            if self.iter_type==(self.numTypes-1):
                if self.initLoss==0:
                    self.initLoss=losses[-1];
                    
                self.prev_lambda_param=self.lambda_param;
                #Choose the value of lambda
                inc_lambda=self.t_eta_param*(losses[-1]-self.lossTh);
                self.lambda_param=self.lambda_param+inc_lambda;
                self.lambda_param=torch.max(self.lambda_param,torch.tensor(0.0,device=losses.device))
                    
                #Update the previous loss
                self.prevLoss=losses[-1].clone();
                #Limit maximum gradient
                tempWeights[-1]=tempWeights[-1]*self.lambda_param#np.minimum(tempWeights[-1],self.lambda_param)
            
            
            #Update paramteres
            if self.iter_type<(self.numTypes-1):
                self.iter_type+=1;  

            #If We are limiting the kl-loss    
            else: 
                #Either we have reached a good solution or found the last iteration
                if losses[-1]<=self.lossTh or self.iter_counter>=self.maxIters:
                    self.iter_counter=0;
                    self.lambda_param=0.0;
                    self.iter_type=0;
                    self.prevLoss=np.inf;
                    self.prev_lambda_param=0;
                    self.t_eta_param=self.eta_param;
                    self.initLoss=0;
                    #Paste the original weights as we will not learn in this step
                    tempWeights=self.weights.clone().detach()
                    updateWeights[...]=0
                     
                else:
                    self.iter_counter+=1
     
        #Apply the weights of each term
        Loss=(tempWeights*losses).sum()
        # print(losses)
        return Loss, updateWeights
    def reset(self):
        self.lambda_param=0.0;
        self.prev_lambda_param=self.lambda_param;
        
    def get_update_weights(self):
         tempWeights=self.weights.clone().detach()
         #Set tempWeights to the current iters
         tempWeights[self.iter_type!=self.types]=0;
         updateWeights=tempWeights>0
         return updateWeights;
