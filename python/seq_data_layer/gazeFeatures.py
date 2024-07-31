# --------------------------------------------------------
# AMT-GAF
# Licensed under BSD-2 License
# Written by Iván González
# --------------------------------------------------------

import numpy as np
import cv2
from sequence_prediction.config import cfg
import pdb
import skimage.io as skio
import os

"""Compute input features using gaze-driven processing:
    seqdb: dataset with inputs
    idxFr: frame list
    step: temporal step: for temporal subsampling
    fmeans: means for feature standardization
    fstds: stds for feature standardization
    idx_perm: array for object permutation (data augmentation)
"""    

def computeGazeFeatures(video,idxFr,step,fmeans,fstds,idx_perm=[]):
    #Parameters
    num_classes = video['data']['scores'].shape[1]
    nFr=int(len(idxFr)/step)
    nMaps=5
    nExtMaps=(nMaps-1)+(num_classes-1)
    nFeatM=nExtMaps
    
    #indexes of the features => For automatic normalization!!
    flimits=[2,2]
    
    #Depending on the dataset, we may have access to IMU data    
    if 'imu' in video['data']:
        nFeatD=10+num_classes+nMaps+(num_classes-1)*2
        flimits.append(6)
    else:
        flimits.append(2)
        nFeatD=6+num_classes+nMaps+(num_classes-1)*2
    
    flimits.extend([5,num_classes,(num_classes-1),(num_classes-1)])
    
    if 'act_score' in video['data']:
        nFeatD=nFeatD+1
        flimits.append(1)
    if 'fix_std' in video['data']:
        nFeatD=nFeatD+2
        flimits.append(2)
        
    flimits=np.array(flimits)
    
    
    center=np.round(video['data']['size']/2);
    maxSize=video['data']['size'].max() 

    
    data_blob = np.zeros((nFr, nFeatD), dtype=np.float32)   
    scores_blob = np.zeros((nFr, num_classes), dtype=np.float32)   
    smap_blob = np.zeros((nFr, nFeatM,cfg.GRID_SIZE[1],cfg.GRID_SIZE[0]), dtype=np.float32)    
    gaze_blob = np.zeros((nFr,cfg.GRID_SIZE[1],cfg.GRID_SIZE[0]), dtype=np.float32)    
    gcoord_blob = np.zeros((nFr, 2), dtype=np.float32)    
    #Visual Working Memory
    VWM=np.zeros((num_classes-1, 3), dtype=np.float32)    

    mapScores = np.zeros((nMaps,), dtype=np.float32)    
    smap_weights_blob = np.zeros((nFr, nMaps), dtype=np.float32)   
    # clips_blob = np.zeros((nFr,1 ), dtype=np.float32)
    time_blob = np.zeros((nFr,), dtype=np.float32)
    
    cf=0;
    contF=0
    
    
    #Loop of frames
    for f in idxFr:
        

        """------------------HOMOGRAPHY COMPUTATION-------------"""
        H=video['data']['homography'][f,...].reshape((3,3))
        #COmpute the inverse homography as it will be needed
        try:
            Hinv=np.linalg.inv(H)
        except:
            H[0,1]=0;
            H[1,0]=0;
            H[2,0:2]=0;
            Hinv=np.linalg.inv(H)
        
        """------------------COMPUTING DYNAMIC FEATURES-------------"""
        fix=video['data']['fix'][f,...];
        if f>0:
            mv_fix=video['data']['mv'][f,...]
            #Computing ego-motion using homography => Project fixation onto the previous frame
            fixh=np.hstack((fix,1))
            fixt=H.dot(fixh);
            fixt=fixt/fixt[2];
            mv_ego=fixt[0:2]-fix
        else:
            mv_fix=np.zeros((2,),dtype=float);
            mv_ego=np.zeros((2,),dtype=float);
        
        """------------------PROCESSING OBJECT SCORES-------------"""
        scores=video['data']['scores'][f,...]
        
        #In case we apply object permutation
        if len(idx_perm)>0:
            scores=scores[idx_perm]
       
        #Dumping the BG score 
        if cfg.BG_SCORE>0:
            scores[0]=scores[0]*cfg.BG_SCORE
        
        #From scores to probs
        scores=np.exp(scores-scores.max())
        scores=scores/scores.sum()
        
        """------------------UPDATING THE VWM-------------"""
        scores_VWM=scores[1:]   
        for c in range(0,num_classes-1):
                
            #Get the previous location and transform it to the current frame
            mem_loc=np.hstack((VWM[c,0:2],1))
            VWM[c,2]=cfg.MEMORY_FACTOR*VWM[c,2]
            
            #Include vWM as input data to make decisions
            #Align the previous detection to be updated
            if VWM[c,2]>0:
                try:
                    curr_loc=Hinv.dot(mem_loc);
                except:
                    pdb.set_trace()
                VWM[c,0:2]=curr_loc[0:2]/curr_loc[2]
             
            #Update the object information with the current detection
            if scores_VWM[c]>cfg.TH_VWM:
                VWM[c,0:2]=(VWM[c,2]*VWM[c,0:2]+scores_VWM[c]*fix)/(VWM[c,2]+scores_VWM[c])
                VWM[c,2]=np.maximum(VWM[c,2],scores_VWM[c])
               
                    
        #If we perform temporal subsampling (step>1)
        if contF%step==0:
            scores_blob[cf,...]=scores
            
            """------------------SETTING-UP THE SPATIAL GRID-------------"""
            qsteps=video['data']['size']/cfg.GRID_SIZE
            xgrid=np.arange(qsteps[0]/2,video['data']['size'][0],qsteps[0]) ;
            ygrid=np.arange(qsteps[1]/2,video['data']['size'][1],qsteps[1]) ;
            xv, yv = np.meshgrid(xgrid, ygrid)
            
            
            """------------------GAZE LABELS-------------"""
            coords=video['data']['fut_gaze'][f,2*(cfg.FRS_TO_PREDICT-1):2*(cfg.FRS_TO_PREDICT)]
            coordsx=coords[0].astype(np.float32)
            coordsy=coords[1].astype(np.float32)
            coordsx=np.clip(coordsx,0,video['data']['size'][0])
            coordsy=np.clip(coordsy,0,video['data']['size'][1])
            gcoord_blob[cf,0]=coordsx/video['data']['size'][0]-0.5
            gcoord_blob[cf,1]=coordsy/video['data']['size'][1]-0.5
            
            cols=np.floor(coordsx/qsteps[0]).astype(int);
            cols=np.maximum(np.minimum(cols,cfg.GRID_SIZE[0]-1),0);
            rows=np.floor(coordsy/qsteps[1]).astype(int);
            rows=np.maximum(np.minimum(rows,cfg.GRID_SIZE[1]-1),0);
            gaze_blob[cf,rows,cols]=1
                
            
            """------------------DYNAMIC FEATURES-------------"""
            iidx=0;
            #Fixation location
            data_blob[cf,iidx]=(np.minimum(np.abs(fix[0]-center[0]),maxSize)-fmeans[0])/fstds[0]
            data_blob[cf,iidx+1]=(np.minimum(np.abs(fix[1]-center[1]),maxSize)-fmeans[0])/fstds[0]
            #indexes of the features => For automatic normalization!!
            
            #Fix-motion
            data_blob[cf,iidx+2]=(np.minimum(np.abs(mv_fix[0]),cfg.MAXMV)-fmeans[1])/fstds[1]
            data_blob[cf,iidx+3]=(np.minimum(np.abs(mv_fix[1]),cfg.MAXMV)-fmeans[1])/fstds[1]
            
            
            #Ego-motion => With and without IMU data
            if 'imu' in video['data']:
                #Subsample IMU as in realtime pupil glasses do not achieve the same rate
                if cf%9==0:
                    data_blob[cf,iidx+4:iidx+10]=(video['data']['imu'][f,...]-fmeans[2])/fstds[2]
                else:
                    data_blob[cf,iidx+4:iidx+10]=data_blob[cf-1,iidx+4:iidx+10]
                iidx+=10;
            else:
                data_blob[cf,iidx+4]=(np.minimum(np.abs(mv_ego[0]),cfg.MAXMV)-fmeans[2])/fstds[2]
                data_blob[cf,iidx+5]=(np.minimum(np.abs(mv_ego[1]),cfg.MAXMV)-fmeans[2])/fstds[2]
                iidx+=6;
            
            #Previous map scores
            data_blob[cf,iidx:iidx+nMaps]=(mapScores-fmeans[3])/fstds[3]
            iidx+=nMaps
            #Object scores
            data_blob[cf,iidx:iidx+num_classes]=(scores-fmeans[4])/fstds[4]
            iidx+=num_classes
           
            #Including the VWM    
            for c in range(0,num_classes-1):
                #Update the object information with the current detection
                data_blob[cf,iidx]=(VWM[c,2]-fmeans[5])/fstds[5]
                #Distance
                data_blob[cf,iidx+num_classes-1]=VWM[c,2]*(np.exp(-np.linalg.norm(VWM[c,0:2]-fix)/(16*(cfg.SIGMA_MAPS**2)))-fmeans[6])/fstds[6]
                iidx+=1
            iidx+=num_classes-1
            
            #Invisible dataset, where AO detectors also provide a guess if the object is being grasped or not
            if 'act_score' in video['data']:
                data_blob[cf,iidx:iidx+1]=(video['data']['act_score'][f]-fmeans[7])/fstds[7]
                iidx+=1
                
            #Invisible dataset    
            if 'fix_std' in video['data']:
                data_blob[cf,iidx:iidx+2]=(video['data']['fix_std'][f]-fmeans[8])/fstds[8]
                
            #Computing the bottom-up saliency map
            frFile=video['images'][f]
            if frFile.find('Frame_')>=0:
                strMask='Mask_%d_%d_'%(cfg.GRID_SIZE[0],cfg.GRID_SIZE[1])
                smFile=frFile.replace('Frame_',strMask).replace('f-',strMask).replace('.jpg','.png');
            elif frFile.find('f-')>=0:
                strMask='m-%d-%d-'%(cfg.GRID_SIZE[0],cfg.GRID_SIZE[1])
                smFile=frFile.replace('f-',strMask).replace('.jpg','.png');
            else:
                strMask='m-%d-%d-'%(cfg.GRID_SIZE[0],cfg.GRID_SIZE[1])
                smFile=frFile.replace('.jpg','_m.png');
            
            #If it is already computed => Just read
            if os.path.exists(smFile):
                saliencyMap=skio.imread(smFile,cv2.IMREAD_GRAYSCALE).astype(float)/255.0;
                if np.isnan(saliencyMap).sum()>0:
                    saliencyMap[...]=1.0
            else:
                im=skio.imread(frFile)
                saliency = cv2.saliency.StaticSaliencySpectralResidual_create()
                (success, saliencyMap) = saliency.computeSaliency(im)
                if np.isnan(saliencyMap).sum()>0:
                    saliencyMap[...]=1.0
                saliencyMap=cv2.resize(saliencyMap,(cfg.GRID_SIZE[0],cfg.GRID_SIZE[1]))
                cv2.imwrite(smFile,(255*saliencyMap).astype(np.uint8))
            

            #Candidate fixation maps
            """------------------SPATIAL MAPS-------------"""
            #Current fixation map => No motion
            smap_blob[cf,0,...]=np.exp(-((xv-fix[0])**2+(yv-fix[1])**2)/cfg.SIGMA_MAPS**2)
            smap_blob[cf,0,...]=smap_blob[cf,0,...]/(smap_blob[cf,0,...].sum()+1.0e-5)
            
            #Center bias
            smap_blob[cf,1,...]=np.exp(-((xv-center[0])**2+(yv-center[1])**2)/(16*cfg.SIGMA_MAPS)**2)
            smap_blob[cf,1,...]=smap_blob[cf,1,...]/(smap_blob[cf,1,...].sum()+1.0e-5)
        
            #Linear motion
            if np.abs(mv_fix).sum()>cfg.TH_MOV:
                fixpred_x=fix[0]+cfg.FRS_TO_PREDICT*mv_fix[0]
                fixpred_y=fix[1]+cfg.FRS_TO_PREDICT*mv_fix[1]
                fixpred_x=np.minimum(np.maximum(fixpred_x,-cfg.SIGMA_MAPS),video['data']['size'][0]+cfg.SIGMA_MAPS)
                fixpred_y=np.minimum(np.maximum(fixpred_y,-cfg.SIGMA_MAPS),video['data']['size'][1]+cfg.SIGMA_MAPS)
                smap_blob[cf,2,...]=np.exp(-((xv-fixpred_x)**2+(yv-fixpred_y)**2)/cfg.SIGMA_MAPS**2)
                smap_blob[cf,2,...]=smap_blob[cf,2,...]/(smap_blob[cf,2,...].sum()+1.3e-5)
            
           
            #Object-based Proposals from working memory
            for c in range(num_classes-1):
                if VWM[c,2]>cfg.TH_VWM:
                    objloc=VWM[c,0:2].astype(int)
                    smap_blob[cf,4+c,...]=np.exp(-((xv-objloc[0])**2+(yv-objloc[1])**2)/cfg.SIGMA_MAPS**2)
                    smap_blob[cf,4+c,...]=smap_blob[cf,4+c,...]/(smap_blob[cf,4+c,...].sum()+1.1e-5)
                    
            #Strong saccades => gaussian with 16std removing the rest applied over a bottom-up saliency map
            smap_blob[cf,3,...]=saliencyMap*np.exp(-((xv-fix[0])**2+(yv-fix[1])**2)/((16*cfg.SIGMA_MAPS)**2))
            smap_blob[cf,3,...]=smap_blob[cf,3,...]/(smap_blob[cf,3,...].sum()+1e-10)
            #Compute accumulated map from other states => Saccades point to unfixed points
            accumulated_smap=smap_blob[cf,0,...]+smap_blob[cf,1,...]+smap_blob[cf,2,...]+smap_blob[cf,4:,...].sum(axis=0)
            smap_blob[cf,3,...]=smap_blob[cf,3,...]-accumulated_smap
            #Renormalization
            smap_blob[cf,3,...]=np.maximum(smap_blob[cf,3,...],0);
            smap_blob[cf,3,...]=smap_blob[cf,3,...]/(smap_blob[cf,3,...].sum()+1e-10)
            
            #Map scores => We can just consider the next frame to be fair
            mapScores[0]=(gaze_blob[cf,...]*smap_blob[cf,0,...]).sum();
            mapScores[1]=(gaze_blob[cf,...]*smap_blob[cf,1,...]).sum();
            mapScores[2]=(gaze_blob[cf,...]*smap_blob[cf,2,...]).sum();
            mapScores[3]=(gaze_blob[cf,...]*smap_blob[cf,3,...]).sum();
            #Object map (VWM)
            omap=smap_blob[cf,4:4+num_classes-1,...].sum(axis=0)
            omap=omap/(omap.sum()+1e-5)
            mapScores[4]=(gaze_blob[cf,...]*omap).sum();
            #Normalization
            mapScores=mapScores/(mapScores.sum()+1e-20);
            
            #Assignment
            smap_weights_blob[cf,...]=mapScores[...]
            """------------------TIME WEIGHTS-------------"""
            time_blob[cf] = float(contF)
            #Internal counter
            cf+=1
            
        #General counter
        contF+=1
    
    return data_blob,gcoord_blob,smap_blob,scores_blob,flimits



