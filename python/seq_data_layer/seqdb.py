# --------------------------------------------------------
# AMT-GAF
# Licensed under BSD-2 License
# Written by Iván González
# --------------------------------------------------------


"""Transform a seqdb into a trainable seqdb by adding a bunch of metadata."""

import numpy as np

def prepare_seqdb(imdb):
    """Prepare the final seqdb with all necessary data: adding the labels
        for each of the videos
    """
    
    seqdb = imdb.seqdb
    
    numFr=0;
    for v in range(len(imdb.video_index)):
        numFr+=seqdb['video_data'][v]['data']['fix'].shape[0]
        
    indexes=np.zeros((numFr,2),dtype=int);
    
    contFr=0;
    maxnFr=0
    #Video index
    for v in range(len(imdb.video_index)):
        seqdb['video_data'][v]['video'] = imdb.video_path_at(v)
        seqdb['video_data'][v]['label'] = imdb.get_actlabel(v)     
        nFr=seqdb['video_data'][v]['data']['fix'].shape[0]
        maxnFr=np.maximum(maxnFr,nFr)
        seqdb['video_data'][v]['images']=[];
        for f in range(nFr):
            seqdb['video_data'][v]['images'].append(imdb.image_path_at(v,f,0))
            indexes[contFr,...]=[v, f];
            contFr+=1;



def get_training_seqdb(imdb):
    
    print('Preparing training data...')
    prepare_seqdb(imdb)
    print('done')
        
    return imdb.seqdb