'''
Created on Jun 2, 2016

@author: isvoboda
'''

import numpy as np
import caffe
import logging
import cv2

from ..utils import decode_dct

module_logger = logging.getLogger(__name__)

class PyDCTVisL(caffe.Layer):
    """
    The dct visualisation layer
    
    Decodes the dct back into the image space a visualize it.
    
    Args:
    -----------
      scale: float
        scale factor of visualized data
             
    Layer prototxt definition:
    --------------------------
    ToDo
    """
    
    def setup(self, bottom, top):
        self.dict_param = dict((key.strip(), val.strip()) for key, val in
                               (item.split(':') for item in
                                self.param_str.split(',')))
        
        if 'scale' in self.dict_param:
            self.scale = int(self.dict_param['scale'])
        else:
            self.scale = 2
        if 'name' in self.dict_param:
            self.name = str(self.dict_param['name'])
        else:
            self.name = "DCT vis"

    def reshape(self, bottom, top):        
            for i_top in xrange(len(top)):
                top[i_top].reshape(1)

    def forward(self, bottom, top):
        img = decode_dct( np.transpose(bottom[0].data[0], [1,2,0]) )
        viz_param = {self.name: img}
        self.visualize(**viz_param)

    def backward(self, top, propagate_down, bottom):
        pass 

    def visualize(self, **kwargs):
#         img_pair = []
        for key, value in kwargs.items():
            preview_resized = cv2.resize(value, (0, 0), fx=self.scale,
                                         fy=self.scale)
            
            cv2.imshow(key, preview_resized/255.)
        cv2.waitKey(5)   
