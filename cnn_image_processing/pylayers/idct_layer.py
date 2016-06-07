'''
Created on Jun 2, 2016

@author: isvoboda
'''

import numpy as np
import caffe
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
    ToDo
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
            dct_diff = code_dct(top[0].diff[i_diff].transpose(1,2,0)) #[y x z]
            bottom[0].diff[i_diff][...] = dct_diff.transpose(2,0,1) #[z y x]
