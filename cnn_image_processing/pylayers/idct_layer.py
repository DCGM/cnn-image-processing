'''
Created on Jun 2, 2016

@author: isvoboda
'''

import numpy as np
import caffe
import cv2
import logging

from ..utils import code_dct
from ..utils import decode_dct

module_logger = logging.getLogger(__name__)

class PyIDCTL(caffe.Layer):
    """
    The idct layer
    
    Decodes the dct back into the image space.
    
    Args:
    -----------
             
    Layer prototxt definition:
    --------------------------
    ToDo Backprop - include the 1024 constant multiplication?
    """
    def setup(self, bottom, top):
        pass
    
    def reshape(self, bottom, top):        
        for i_top in xrange(len(top)):
            shape = np.asarray(bottom[i_top].data.shape)
            shape[2:] = shape[2:]*8
            shape[1] = 1
            top[i_top].reshape(*shape)

    def forward(self, bottom, top):
        for i_data in xrange(len(bottom[0].data)):
            coefs = np.transpose(bottom[0].data[i_data], [1,2,0]) #[y x z]
            top[0].data[i_data, ...] = decode_dct(coefs).transpose([2,0,1]) #[z y x]
        
    def backward(self, top, propagate_down, bottom):           
        for i_diff in xrange(len(top[0].diff)):
            top_diff = top[0].diff[i_diff].transpose([1,2,0]) #[y x z]
            bottom_data = bottom[0].data[i_diff].transpose([1,2,0]) #[y x z]
            bottom_diff = bottom[0].diff[i_diff]
            step = 8
            for y_coef in xrange(bottom_data.shape[0]):
                for x_coef in xrange(bottom_data.shape[1]):
                    top_x = step*x_coef
                    top_y = step*y_coef
                    diff_block = top_diff[top_y:top_y+step,top_x:top_x+step]
                    # 1. Compute derivative of f: idct(x) / d_bottom
                    d_bottom_d_top = cv2.dct(diff_block)
                    d_bottom_d_top = d_bottom_d_top.reshape(64)
                    bottom_diff[:, y_coef, x_coef] = d_bottom_d_top
