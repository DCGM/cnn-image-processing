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
        pass

    def reshape(self, bottom, top):        
            for i_top in xrange(len(top)):
                top[i_top].reshape(1)

    def forward(self, bottom, top):    
        pass

    def backward(self, top, propagate_down, bottom):
        pass
    
      
    def visualize(self, **kwargs):
#         img_pair = []
        for key, value in kwargs.items():
            img = value.transpose(1, 2, 0) / self.normalize
            img += self.vis_mean
#             img_pair.append(img)
            preview_resized = cv2.resize(img, (0, 0), fx=self.vis_scale,
                                         fy=self.vis_scale)
            
            cv2.imshow(key, preview_resized/255.)
#         cv2.imshow(key, np.vstack(img_pair)/255.)
        cv2.waitKey(5)   
