'''
Created on May 27, 2016

@author: isvoboda
'''

import numpy as np
import caffe
import logging
import cv2

module_logger = logging.getLogger(__name__)

class PyVisLayer(caffe.Layer):
    """
    The visualisation layer
    Args:
    -----------
      
      scale: float
        scale factor of visualized data
      
      normalize: float
        normalization factor
        
      mean: float
        mean added to the visualized data
      
        
    Layer prototxt definition:
    --------------------------
    ToDo
    """
    
    def setup(self, bottom, top):
        self.dict_param = dict((key.strip(), val.strip()) for key, val in
                               (item.split(':') for item in
                                self.param_str.split(',')))
        
        if 'scale' in self.dict_param:
            self.scale = float(self.dict_param['scale'])
        else:
            self.scale = 1
         
        if 'mean' in self.dict_param:
            self.mean = float(self.dict_param['mean'])
        else:
            self.mean = 0
         
        if 'norm' in self.dict_param:
            self.norm = float(self.dict_param['norm'])
        else:
            self.norm = 1
        
        if 'name' in self.dict_param:
            self.name = str(self.dict_param['name'])
        else:
            self.name = "Vis"

    def reshape(self, bottom, top):        
            for i_top in xrange(len(top)):
                top[i_top].reshape(1)

    def forward(self, bottom, top):
        vis_data = bottom[0].data[0]
        viz_param = {self.name: vis_data}
        self.visualize(**viz_param)
        
        for i_data in xrange(len(top)):
            top[i_data].data[...] = bottom[i_data].data

    def backward(self, top, propagate_down, bottom):
        for i_diff in xrange(len(top)):
            bottom[i_diff].diff[...] = top[i_diff].diff
      
    def visualize(self, **kwargs):
        for key, value in kwargs.items():
            img = value.transpose(1, 2, 0) / self.norm # [y x z]
            img += self.mean
            preview_resized = cv2.resize(img, (0, 0), fx=self.scale,
                                         fy=self.scale)
            
            cv2.imshow(key, preview_resized)
        cv2.waitKey(5)   
