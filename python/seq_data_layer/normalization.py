# --------------------------------------------------------
# AMT-GAF
# Licensed under BSD-2 License
# Written by Iván González
# --------------------------------------------------------

import numpy as np
import torch
import pdb

"""Function that computes 1st and 2nd order moments considering the limits
of the features in the feature array"""
def updateMoms(data,mom1,mom2,flimits):
    numF=len(flimits)
    idxc=0
    for i in range(numF):
        idx=range(idxc,idxc+flimits[i]);
        mom1[i]+=data[...,idx].mean()
        mom2[i]+=(data[...,idx]**2).mean()
        idxc=flimits[i]
      
    return mom1,mom2

def computeNorms(mom1,mom2,contIters):
    mom1=mom1/contIters
    mom2=mom2/contIters
    means=mom1;
    stds=torch.sqrt(mom2-mom1**2)
    print(means)
    print(stds)
    return means,stds