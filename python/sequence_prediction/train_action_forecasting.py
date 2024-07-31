# --------------------------------------------------------
# AMT-GAF
# Licensed under BSD-2 License
# Written by Iván González
# --------------------------------------------------------

"""Train a Fast R-CNN network."""
import warnings
warnings.filterwarnings("ignore")

from sequence_prediction.config import cfg
from utils.timer import Timer
import numpy as np
import os
import pdb
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.nn as nn
from losses.action_prediction_losses import WeakSeqLossLayer
from losses.asymmetric_loss import AsymmetricLoss
from performance_metrics.metrics import SeqAcc
from performance_metrics.metrics import ComputePrecRecall
from torch.nn.utils import clip_grad_norm_
import seqmodels.amt_gaf as sq
from seq_data_layer.normalization import computeNorms,updateMoms

def check_snapshots(output_dir,num_epochs):
    for epoch in range(num_epochs,-1,-1):
        if os.path.isfile(output_dir + '/model_epoch%d.pth'%epoch):
            return epoch
    return -1

def load_checkpoint(pmodel, optimizer, lr_sched, device,state_filename,model_filename):
    # Note: Input model & optimizer should be pre-defined.  This routine only updates their states.
    start_epoch = 0
    if os.path.isfile(state_filename):
        
        print("=> loading checkpoint '{}' & '{}'".format(state_filename,model_filename))
        checkpoint = torch.load(state_filename)
        start_epoch = checkpoint['epoch']+1
        
        optimizer.load_state_dict(checkpoint['optimizer'])
        if device.type=="cuda":
            for state in optimizer.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.cuda()
        lr_sched=checkpoint['lr_sched']
        pmodel.load_state_dict(torch.load(model_filename))
        #Convert net to device
        pmodel=pmodel.to(device)
        print("=> loaded checkpoint '{}' & '{}' (epoch {})"
                  .format(state_filename, model_filename,checkpoint['epoch']))
    else:
        print("=> no checkpoint found at '{}'. Just reading model '{}'".format(state_filename,model_filename))
        pmodel.load_state_dict(torch.load(model_filename),strict=False)
        #Convert net to device
        pmodel=pmodel.to(device)
        
    return pmodel, optimizer, start_epoch, lr_sched

#Function that implements the training process`
def train_net(seqDataset, output_dir,pretrained_model=None, epochs=25):
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    #Get Feat dimensions
    sample=seqDataset[0]
    featLength=sample['data'].shape[1]
    
    
    num_objclasses=seqDataset.num_objclasses
    num_actclasses=seqDataset.num_actclasses

    """Create the model"""
    #Model parameters
    nMap=int(sample['smap'].shape[1])
    height,width=sample['smap'].shape[2:]
    #Create the model
    pmodel=sq.AMTGAF(featLength,cfg.ATT_SIZE,cfg.ZLENGTH,num_actclasses,cfg.GRID_SIZE,nMap=nMap,mapOp=cfg.MAP_OP,dropout=cfg.DROPOUT,temporal_horizon=cfg.TEMPORAL_HORIZON)
    #Convert net to device
    pmodel=pmodel.to(device)

    """Training objects"""
    #Data loader
    dataloader = DataLoader(seqDataset, batch_size=cfg.TRAIN.BATCH_SIZE,shuffle=True,num_workers = cfg.TRAIN.NUM_WORKERS,pin_memory=True)
    # Optimizer
    optimizer = optim.SGD(pmodel.parameters(), momentum=cfg.TRAIN.MOMENTUM, lr=cfg.TRAIN.LR, weight_decay=cfg.TRAIN.WD)
    # LR scheduler
    lr_sched = lr_scheduler.StepLR(optimizer, cfg.TRAIN.LR_STEP_SIZE, cfg.TRAIN.LR_GAMMA)
    
    """Loss definition"""
    ActLoss=WeakSeqLossLayer(num_actclasses,preFrames=cfg.TRAIN.ACTLOSS_PRE_FRAMES, minFramesDet=cfg.TRAIN.ACTLOSS_MIN_FRAMES_DET,mul=cfg.TRAIN.ACTLOSS_MUL)
        
    weights=torch.tensor(cfg.TRAIN.LOSS_WEIGHTS,dtype=torch.float32,device=device,requires_grad=False)
    GazeLoss=nn.MSELoss()        
    types=torch.tensor([0,1],device=device,requires_grad=False).int()
    MTLoss=AsymmetricLoss(weights=weights, types=types, lossTH=cfg.TRAIN.LOSS_TH, eta_param=cfg.TRAIN.LOSS_ETA,maxIters=cfg.TRAIN.LOSS_MAX_ITERS )
    
    """Initialization of performance metrics"""
    timer = Timer()
    tr_loss=np.zeros(epochs+1,)
    tr_actloss=np.zeros(epochs+1,)
    tr_gazeloss=np.zeros(epochs+1,)
    
    numCortes=len(cfg.TIME_MARKS);
    
    Accuracy=SeqAcc(num_obj_classes=num_objclasses,num_act_classes=num_actclasses,
                    minFramesDet=cfg.TRAIN.ACC_MIN_FRAMES_DET,framesBetween=cfg.TRAIN.ACC_FRAMES_BETWEEN,
                    preFrames=cfg.TRAIN.ACC_PRE_FRAMES)
    prec=np.zeros((epochs+1,numCortes),dtype=float)
    recall=np.zeros((epochs+1,numCortes),dtype=float)
    Fscore=np.zeros((epochs+1,numCortes),dtype=float)
    numCortes=len(cfg.TIME_MARKS);
    
    
    """Model resuming"""
    init_epoch=0
    #Check if we can resume
    last_epoch=check_snapshots(output_dir,epochs)
    #If we resume load the model and previous statistics
    if last_epoch>=0:
        print("Resuming from epoch %d"%last_epoch)
        model_filename=output_dir + '/model_epoch%d.pth'%last_epoch
        state_filename=output_dir + '/train_state.pth'
        
        pmodel, optimizer, init_epoch, lr_sched =load_checkpoint(pmodel, optimizer, lr_sched, device,state_filename,model_filename)   
        lossFile=os.path.join(output_dir, 'train_loss.txt');
        with open(lossFile) as f: # open the file for reading
            for line in f: # iterate over each line
                epoch,actloss, gazeloss,loss,F1,F2=line.split()
                epoch=int(epoch);
                if epoch<=last_epoch:
                    tr_actloss[epoch] = float(actloss) # convert age from string to int
                    tr_gazeloss[epoch] = float(gazeloss)
                    tr_loss[epoch] = float(loss)
                    Fscore[epoch,0] = float(F1) # convert bs from string to float
                    Fscore[epoch,numCortes-1] = float(F2)
        init_epoch=last_epoch+1
    
    #If we do automatic per-video normalization instead of corpus-level normalization
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
        
    """Network training loop."""
    timer = Timer()
    for epoch in range(init_epoch,epochs):
        
        contIters=0
        itersEpoch=len(seqDataset)/cfg.TRAIN.BATCH_SIZE+1
        
        tp=np.zeros((num_actclasses,0),dtype=float)
        fp=np.zeros((num_actclasses,1),dtype=float)
        pos=np.zeros((num_actclasses,1),dtype=float)
    
        for sample in dataloader:
            timer.tic()
            
            optimizer.zero_grad()
            
            """Transforming data to feed the model"""
            scores=sample['scores'].to(device)
            data=sample['data'].to(device)
            smap=sample['smap'].to(device)
            time=sample['time'].to(device)
            labels=sample['labels'].to(device)
            
            int_iter=0
            """Loop for constrained optimization problem"""
            while 1:
                
                #Set gradients to zero in the optimizer
                optimizer.zero_grad()
                #Forward pass
                outs = pmodel(data,smap)
                
                #Get the outputs
                outputs = outs['act_preds']
                fixations = outs['fixations']
                
                #Compute the Gaze loss
                gt_fix=sample['gaze'].to(device)
                gLoss=2*GazeLoss(fixations,gt_fix)
                    
                #Compute the Action Prediction Loss
                actLoss=ActLoss(outputs,labels,time,scores)
                
                #Compute the Multi-task loss
                losses=torch.stack((actLoss, gLoss),dim=0)
                totalLoss, updateWeights=MTLoss(losses,initialize=(int_iter==0))
                
                #Values for statistics (get the first iter)
                if int_iter==0:
                    show_outputs=outputs.clone()
                    show_actLoss=actLoss.clone().detach()
                    show_gLoss=gLoss.clone().detach()
                #Check convergence
                converged=updateWeights.sum()==0
                if converged:
                    #Values for statistics
                    gLoss=show_gLoss;
                    actLoss=show_actLoss
                    gLoss=show_gLoss;
                    actLoss=show_actLoss
                    totalLoss=weights@(torch.tensor((show_actLoss,show_gLoss),device=device))
                    outputs=show_outputs;
                    
                    break;
                else:
                    #Run the backward step to learn
                    totalLoss.backward()
                    if cfg.TRAIN.CLIP_GRADS>0:
                        clip_grad_norm_(pmodel.parameters(), cfg.TRAIN.CLIP_GRADS)
                    #Update of the optimizer
                    optimizer.step()
                    int_iter+=1;

            timer.toc()
            
            #Compute performance metrics 
            with torch.no_grad():
                #Action prediction Accuracy
                ttp,tfp,tpos = Accuracy(outputs,labels,time,scores)
                tp=np.hstack((tp,ttp.cpu().numpy()));
                fp+=tfp.cpu().numpy()
                pos+=tpos.cpu().numpy();
                
            showIter=int(np.ceil(itersEpoch/10))
            if contIters%showIter==0:
                print('it %d/%d'%(contIters,itersEpoch))
            contIters+=1    
            
            #Update losses
            tr_gazeloss[epoch]+=gLoss
            tr_actloss[epoch]+=actLoss
            tr_loss[epoch]+=totalLoss
        
        #Update the lr scheduler
        lr_sched.step()    
        
        #End of an epoch - Compute and show global statistics 
        #Losses                
        tr_loss[epoch]=tr_loss[epoch]/contIters
        tr_actloss[epoch]=tr_actloss[epoch]/contIters
        tr_gazeloss[epoch]=tr_gazeloss[epoch]/contIters
        # Action prediction statistics
        (tprec,trecall)=ComputePrecRecall(tp,fp,pos,cfg.TIME_MARKS);
        tF=2*(tprec*trecall)/(tprec+trecall+1e-10)
        prec[epoch,...]=tprec[...];
        recall[epoch,...]=trecall[...];
        Fscore[epoch,...]=tF[...];
                
        print('========================================================================')
        print('epoch %d GP: act-loss %.3f gaze-loss %.3f Total Loss: %.3f'%(epoch,tr_actloss[epoch],tr_gazeloss[epoch],tr_loss[epoch]))
        np.set_printoptions(precision=3,linewidth=400)
        print('precs: ' + np.array2string(prec[epoch,...], formatter={'float_kind':lambda x: "%.2f" % x}) + ' %.2f'%(prec[epoch,...].mean()))                       
        print('recalls: ' + np.array2string(recall[epoch,...], formatter={'float_kind':lambda x: "%.2f" % x}) + ' %.2f'%(recall[epoch,...].mean()))
        print('F-measures: ' + np.array2string(Fscore[epoch,...], formatter={'float_kind':lambda x: "%.2f" % x}) + ' %.2f'%(Fscore[epoch,...].mean()))
        print('speed: {:.3f}s / iter'.format(timer.average_time))

        #Figures and plots
        fig1 = plt.gcf()
        plt.plot(range(epoch+1),tr_actloss[0:epoch+1],'g')
        plt.plot(range(epoch+1),tr_gazeloss[0:epoch+1],'c')
        plt.plot(range(epoch+1),tr_loss[0:epoch+1],'b')
        plt.ylabel('Training loss')
        plt.legend(('act-loss','gaze-loss','Total Loss'),loc=1)
        fig1.savefig(os.path.join(output_dir, 'train_loss.png'))
        plt.close('all')  
        
        fig1 = plt.gcf()
        plt.plot(range(epoch+1),Fscore[0:epoch+1,0],'g')
        plt.plot(range(epoch+1),Fscore[0:epoch+1,numCortes-1],'b')
        plt.ylabel('Performance')
        plt.legend(('AUC','F-fast','F-slow'),loc=4)
        fig1.savefig(os.path.join(output_dir, 'train_perf.png'))
        plt.close('all')  
        
        outLogFile=os.path.join(output_dir, 'train_loss.txt')
        np.savetxt(outLogFile,np.vstack((range(epoch+1),tr_actloss[0:epoch+1], tr_gazeloss[0:epoch+1],tr_loss[0:epoch+1],Fscore[0:epoch+1,0],Fscore[0:epoch+1,numCortes-1])).transpose(),"%d %.3f %.3f %.3f %.3f %.3f")    
        
        #Every 5 epochs we save a snapshot in the folder
        if epoch%5==0:
            checkpoint = { 
                'epoch': epoch,
                'optimizer': optimizer.state_dict(),
                'lr_sched': lr_sched
                }
            torch.save(checkpoint, output_dir + '/train_state.pth')
            torch.save(pmodel.state_dict(), output_dir + '/model_epoch%d.pth'%epoch)
        
