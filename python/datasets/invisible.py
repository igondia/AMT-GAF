# --------------------------------------------------------
# AMT-GAF
# Licensed under BSD-2 License
# Written by Iván González
# --------------------------------------------------------

import datasets
import datasets.invisible
import os
import datasets.imdb
import numpy as np
import scipy.sparse
import pdb
from sequence_prediction.config import cfg


class invisible(datasets.imdb):
    
    def __init__(self, image_set, fold=0, db_dir=None):
        
        if fold>0:
            datasets.imdb.__init__(self, 'invisible_' + image_set + '_' + '%d'%fold)
        else:
            datasets.imdb.__init__(self, 'invisible_' + image_set)
       
        
        self._fold = fold
        self._image_set = image_set
        """I follow pascal terminology, but in our case devkit and datafolders are the same"""
        
        self._db_dir = self._get_default_path() if db_dir is None \
                            else db_dir
        self._data_path = self._db_dir #os.path.join(self._db_dir, 'DATA')
        self._frame_pattern = ''
        self._frames_subfolder = 'Frames'
        self._aug_frames_subfolder = 'FramesAug'
        self._ann_subfolder = 'annotations'
        self._frame_step = 1
        self._num_objlabels_per_video=1
        
        """Read the classes from the corresponding file"""
        catFile=os.path.join(self._db_dir,'categories.npy')
        categories=list(np.load(catFile,allow_pickle=True))
        
        
        self._objclasses = tuple(categories)
        self._objclass_to_ind = dict(zip(self.objclasses, range(self.num_objclasses)))
        #The last one is the no target        
        self._no_tgt_category = self._objclass_to_ind['no_targets']
        self._verbclasses = ('view','grasp')
        
        act_classes=[]#['no_action']
        for obj in self._objclasses:
            for act in self._verbclasses:
                act_classes.append(act + '_' + obj)
        act_classes.pop(act_classes.index('grasp_no_targets'))
        act_classes[act_classes.index('view_no_targets')]='no_targets'
        
        self._actclasses = tuple(act_classes)
        self._actclass_to_ind = dict(zip(self.actclasses, range(self.num_actclasses)))
        
        self._image_ext = '.jpg'
        self._ann_ext = '.png'
        self._video_index, self._objlabel, self._actlabel = self._load_video_set_index()
        
        
        
            
        #Default to seq_handler
        self._seqdb_handler = self.fix_objects_seqdb
        
        assert os.path.exists(self._db_dir), \
                'invisible path does not exist: {}'.format(self._db_dir)
        assert os.path.exists(self._data_path), \
                'Data Path does not exist: {}'.format(self._data_path)

        
    def get_objlabel(self, v):
        """
        Return the label number for the video.
        """
        label = self._objlabel[v]
        return label
    
    def get_actlabel(self, v):
        """
        Return the label number for the video.
        """
        label = self._actlabel[v]
        return label

    def video_path_at(self, i):
        """
        Return the absolute path to image i in the image sequence.
        """
        return self.video_path_from_index(self._video_index[i])

    def video_path_from_index(self, index):
        """
        Construct an image path from the image's "index" identifier.
        """
        video_path = os.path.join(self._data_path, 'images',
                                  index )
        #assert os.path.exists(video_path), \
        #        'Path does not exist: {}'.format(video_path)
        if os.path.exists(video_path)==0:
            print('Path does not exist: %s'%video_path)
        return video_path
    
    def image_path_at(self, v, i, nv):
        """
        Return the absolute path to image i in the image sequence.
        """
        return self.image_path_from_index(self._video_index[v], i, nv)
    
    def image_path_from_index(self, vindex, index, nvar):
        """
        Construct an image path from the image's "index" identifier.
        """
        if nvar == 0:
            image_path = os.path.join(self._data_path, 'images',
                                  vindex, self._frames_subfolder, '%s%d%s'%(self._frame_pattern,index*self._frame_step,self._image_ext) )
        else:
            image_path = os.path.join(self._data_path, 'images',vindex, self._aug_frames_subfolder, '%s%d_v%d%s'%(self._frame_pattern,index*self._frame_step,nvar,self._image_ext) )
                
                

        return image_path
    
    def annotation_path_at(self, v, i):
        """
        Return the absolute path to image i in the image sequence.
        """
        return self.annotation_path_from_index(self._video_index[v], i)
    
    def annotation_path_from_index(self, vindex, index):
        """
        Construct an image path from the image's "index" identifier.
        """
        ann_path = os.path.join(self._data_path, self._ann_subfolder,
                                  vindex, '%s%05d%s'%(self._frame_pattern,index*self._frame_step,self._ann_ext) )
        
        return ann_path 
    
    def _load_video_set_index(self):
        """
        Load the indexes listed in this dataset's image set file.
        """
        # Example path to image set file:
        # self._db_dir + /VOCdevkit2007/VOC2007/ImageSets/Main/val.txt
        if self._fold>0:
            video_set_file = os.path.join(self._data_path, 'ImageSets', self._image_set + '%d'%self._fold + '.txt')
        else:
            video_set_file = os.path.join(self._data_path, 'ImageSets', self._image_set + '.txt')
            
        
        
        assert os.path.exists(video_set_file ), \
                'Path does not exist: {}'.format(video_set_file )
       
        with open(video_set_file ) as f:
            video_data = [x.strip() for x in f.readlines()]
        video_index = []
        obj_labels = []
        act_labels = []
        for vdata in video_data:
            tdata=vdata.split()
            video_index.append(' '.join(tdata[0:-1]))
            obj_labels.append(int(tdata[-1]))
            
            if tdata[0].find('grasp')>=0:
                act_labels.append(int(1))
            else:
                act_labels.append(int(0))
        return video_index,obj_labels,act_labels
    

    def _get_default_path(self):
        """
        Return the default path where Grasp in the Wild is expected to be installed.
        """
        return cfg.DB_DIR

        
    def fix_objects_seqdb(self):
        """
        Return the database of fixations plus objects.

        """
        
        filename = os.path.abspath(os.path.join(self._data_path,
                                                'seq',
                                                self.seqname + '.npy'))
        
        assert os.path.exists(filename), \
               'Seq data not found at: {}'.format(filename)
        data=np.load(filename,allow_pickle=True)
        
        return self.create_seqdb_from_seqdata(data)

  
  
    def set_seq_db(self, seqname):
        self.seqname = seqname
    
    
    #Function that creates the seq db reading the corresponding files
    def create_seqdb_from_seqdata(self,data):
        video_data = []
        for v in range(self.num_videos):
            video = {'fix' : data[v]['X1'][...,0:2],
                     'fix_std' : data[v]['X1'][...,2:4],
                     'mv' : data[v]['X1'][...,4:6],
                     'imu' : data[v]['X1'][...,6:12],
                     'size': data[v]['X1'][0,12:14],
                     'homography' : data[v]['X1'][...,14:23],
                     'labels' : data[v]['Y'][...],
                     'scores': data[v]['X1'][...,23:-1],
                     'act_score': data[v]['X1'][...,-1],
                     # 'hand_score': data[v]['X1'][...,-2:],
                     'fut_gaze': data[v]['G'][...]}
            if(video['labels']>=0).sum()==0:
                print('Video %d has no frames'%v)
            video_data.append({'data' : video})
            
        seqdb= {'video_data' : video_data,'indexes' : [],'onet' : None};
        return seqdb
        
   