# --------------------------------------------------------
# AMT-GAF
# Licensed under BSD-2 License
# Written by Iván González
# --------------------------------------------------------

import datasets
import datasets.sharon
import os
import datasets.imdb
import numpy as np
from sequence_prediction.config import cfg


class sharon(datasets.imdb):
    
    def __init__(self, image_set, fold=0, db_dir=None):
        
        if fold>0:
            datasets.imdb.__init__(self, 'sharon_' + image_set + '_' + '%d'%fold)
        else:
            datasets.imdb.__init__(self, 'sharon_' + image_set)
       
        
        self._fold = fold
        self._image_set = image_set
        """I follow pascal terminology, but in our case devkit and datafolders are the same"""
        
        self._db_dir = self._get_default_path() if db_dir is None \
                            else db_dir
        self._data_path = self._db_dir 
        self._frame_pattern = 'f-'
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
        self._actclasses = tuple(categories)
        self._actclass_to_ind = dict(zip(self.actclasses, range(self.num_actclasses)))
        self._verbclasses = ('np_grasp','grasp')
        
        self._image_ext = '.jpg'
        self._ann_ext = '.png'
        self._video_index, self._objlabel, self._actlabel = self._load_video_set_index()
        
        #Default to seq_handler
        self._seqdb_handler = self.fix_objects_seqdb
        
        assert os.path.exists(self._db_dir), \
                'Sharon path does not exist: {}'.format(self._db_dir)
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
                                  vindex, self._frames_subfolder, '%s%05d%s'%(self._frame_pattern,index*self._frame_step,self._image_ext) )
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
        for vdata in video_data:
            tdata=vdata.split()
            video_index.append(' '.join(tdata[0:-1]))
            obj_labels.append(int(tdata[-1]))
        act_labels=obj_labels.copy()
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
                     'mv' : data[v]['X1'][...,2:4],
                     'imu' : data[v]['X1'][...,4:10],
                     'size': data[v]['X1'][0,10:12],
                     'homography' : data[v]['X1'][...,12:21],
                     'labels' : data[v]['Y'][...],
                     'scores': data[v]['X1'][...,21:],
                     'fut_gaze': data[v]['G'][...]}
                         
            video_data.append({'data' : video})
            
        seqdb= {'video_data' : video_data,'indexes' : [],'onet' : None};
        return seqdb
             
        
