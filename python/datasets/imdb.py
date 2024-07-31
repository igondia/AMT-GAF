# --------------------------------------------------------
# AMT-GAF
# Licensed under BSD-2 License
# Written by Iván González
# --------------------------------------------------------

import os
import os.path as osp
import PIL
import numpy as np
import scipy.sparse
import pdb

class imdb(object):
    """Image database."""

    def __init__(self, name):
        self._name = name
        self._bbname = name
        self._seqname = name
        self._num_objclasses = 0
        self._objclasses = []
        self._num_actclasses = 0
        self._actclasses = []
        
        self._video_index = []
        self._obj_proposer = 'selective-search'
        self._roidb = None
        self._seqdb = None
        self._roidb_handler = self.default_roidb
        self._seqdb_handler = self.default_seqdb
        # Use this dict for storing dataset specific config options
        self.config = {}
        self._numVar = 1
        self.onet = None
        
    @property
    def name(self):
        return self._name
  
    @property
    def num_objclasses(self):
        return len(self._objclasses)
    
    @property
    def objclasses(self):
        return self._objclasses

    @property
    def num_verbclasses(self):
        return len(self._verbclasses)
    
    @property
    def verbclasses(self):
        return self._verbclasses

    @property
    def num_actclasses(self):
        return len(self._actclasses)

    @property
    def actclasses(self):
        return self._actclasses
    
    @property
    def video_index(self):
        return self._video_index

    @property
    def roidb_handler(self):
        return self._roidb_handler
    
    @property
    def seqdb_handler(self):
        return self._seqdb_handler

    @roidb_handler.setter
    def roidb_handler(self, val):
        self._roidb_handler = val

    @seqdb_handler.setter
    def seqdb_handler(self, val):
        self._seqdb_handler = val
        
    @property
    def roidb(self):
        # A roidb is a list of dictionaries, each with the following keys:
        #   boxes
        #   gt_overlaps
        #   gt_classes
        #   flipped
        if self._roidb is not None:
            return self._roidb
        self._roidb = self.roidb_handler()
        return self._roidb

    @property
    def seqdb(self):
        if self._seqdb is not None:
            return self._seqdb
        self._seqdb = self.seqdb_handler()
        return self._seqdb
    
    
    @property
    def num_videos(self):
      return len(self.video_index)

    def image_path_at(self, i):
        raise NotImplementedError

    def default_roidb(self):
        raise NotImplementedError

    def default_seqdb(self):
        raise NotImplementedError
        
    def evaluate_detections(self, all_boxes, output_dir=None):
        """
        all_boxes is a list of length number-of-classes.
        Each list element is a list of length number-of-images.
        Each of those list elements is either an empty list []
        or a numpy array of detection.

        all_boxes[class][image] = [] or np.array of shape #dets x 5
        """
        raise NotImplementedError

    def append_flipped_images(self):
        num_videos = self.num_videos
        widths = [PIL.Image.open(self.image_path_at(i)).size[0]
                  for i in range(num_videos)]
        for i in range(num_videos):
            boxes = self.roidb[i]['boxes'].copy()
            oldx1 = boxes[:, 0].copy()
            oldx2 = boxes[:, 2].copy()
            boxes[:, 0] = widths[i] - oldx2 - 1
            boxes[:, 2] = widths[i] - oldx1 - 1
            assert (boxes[:, 2] >= boxes[:, 0]).all()
            entry = {'boxes' : boxes,
                     'gt_overlaps' : self.roidb[i]['gt_overlaps'],
                     'gt_classes' : self.roidb[i]['gt_classes'],
                     'flipped' : True}
            self.roidb.append(entry)
        self._video_index = self._video_index * 2


    def create_roidb_from_box_list(self, box_list, gt_roidb):
        assert len(box_list) == self.num_videos, \
                'Number of boxes must match number of ground-truth images'
        roidb = []
        for v in range(self.num_videos):
            posFr=0
            nFr = len(box_list[v])
            video = []
            for f in range(nFr): 
                boxes = box_list[v][f]

                #If we have a positive frame
                if boxes[...,0].sum()>0:
                    posFr=1
                else:
                    #Don't consider negatives after the positive
                    if posFr==1:
                        break
                video.append({'boxes' : boxes[...,1:],
                              'fg' : boxes[...,0],
                              'flipped' : False})     
            roidb.append({'frames' : video})      
            
        self.set_bb_numVar(box_list[0][0].shape[0])
        return roidb
    
    def create_multiclass_roidb_from_box_list(self, box_list, gt_roidb):
        assert len(box_list) == self.num_videos, \
                'Number of boxes must match number of ground-truth images'
        roidb = []
        ant_labels = np.zeros_like(box_list[0][0][0,0,5:],dtype=int)
        
        for v in range(self.num_videos):
            posFr=0
            seq = 0
            ant_labels[...] = -1;
            nFr = len(box_list[v])
            video = []
            for f in range(nFr): 
                boxes = box_list[v][f]

                #Check if we are on a different sequence
                labels = boxes[0,0,5:];
                
                    
                #If we have a positive frame
                if boxes[...,0].sum()>0:
                    posFr=1
                    if (labels!=ant_labels).sum()>0:
                        seq = seq + 1;
                        if labels.sum()==0:
                            pdb.set_trace()
                    ant_labels[...]=labels
                else:
                    posFr=0;
                    ant_labels[...]=-1;
                    
                
                seq_fr=seq*posFr
                
                video.append({'boxes' : boxes[...,1:5],
                              'fg' : boxes[...,0],
                              'labels' : labels,
                              'seq' : seq_fr,
                              'flipped' : False})     
            roidb.append({'frames' : video})      
            
        self.set_bb_numVar(box_list[0][0].shape[0])
        return roidb
        
    #Function that creates the seq db reading the corresponding files
#    def create_seqdb_from_seqdata(self,data):
#        
#        video_data = []
#        for v in xrange(self.num_videos):
#            video = {'fix' : data[v][...,0:2],
#                     'mv' : data[v][...,2:4],
#                         'size': data[v][0,4:6],
#                          'labels' : data[v][...,6],
#                          'scores': data[v][...,7:]}
#            video_data.append({'data' : video})      
#        seqdb= {'video_data' : video_data,'indexes' : []};
#        return seqdb
    
    #Function that creates the seq db reading the corresponding files
    def create_seqdb_from_seqdata(self,data):
        
        video_data = []
        for v in range(self.num_videos):
            video = {'fix' : data[v]['X1'][...,0:2],
                     'mv' : data[v]['X1'][...,2:4],
                     'size': data[v]['X1'][0,4:6],
                     'homography' : data[v]['X1'][...,6:15],
                     'labels' : data[v]['Y'][...],
                     'scores': data[v]['X1'][...,15:],
                     'fut_gaze': data[v]['G'][...]}
            if data[v]['Y'][...].sum()==0:
                pdb.set_trace()
                         
            video_data.append({'data' : video})
            
        seqdb= {'video_data' : video_data,'indexes' : [],'onet' : None};
        return seqdb
        
    @staticmethod
    def merge_roidbs(a, b):
        assert len(a) == len(b)
        for i in range(len(a)):
            a[i]['boxes'] = np.vstack((a[i]['boxes'], b[i]['boxes']))
            a[i]['gt_classes'] = np.hstack((a[i]['gt_classes'],
                                            b[i]['gt_classes']))
            a[i]['gt_overlaps'] = scipy.sparse.vstack([a[i]['gt_overlaps'],
                                                       b[i]['gt_overlaps']])
        return a

    def competition_mode(self, on):
        """Turn competition mode on or off."""
        pass

    def set_bb_numVar(self,numVar):
        self._numVar=numVar
