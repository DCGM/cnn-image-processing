from __future__ import print_function
from __future__ import division

import numpy as np
import caffe
import logging

class PyAddL(caffe.Layer):
    """A layer that compute bottom[0].data + bottom[1].data"""
    
    def setup(self, bottom, top):
        if len(bottom) != 2 :
            raise Exception("PySubL has to have 2 inputs.")
        self.i_data = 0
        self.i_label = 1
        # Compute the border
        label_shape = np.asarray(bottom[self.i_label].data.shape[2:])
        data_shape = np.asarray(bottom[self.i_data].data.shape[2:])
        self.borders = label_shape - data_shape
        self.borders //= 2
        
    def reshape(self, bottom, top):
        top[0].reshape(*bottom[self.i_data].data.shape)
    
    def forward(self, bottom, top):
        (i_crop_x, i_crop_y) = self.borders
        len_x = bottom[self.i_data].data.shape[2]
        len_y = bottom[self.i_data].data.shape[3]
        label_data = bottom[self.i_label].data
        crop_data = label_data[:, :, i_crop_x:i_crop_x + len_x,
                               i_crop_y:i_crop_y + len_y]
        top[0].data[...] = crop_data + bottom[self.i_data].data
    
    def backward(self, top, propagate_down, bottom):
        bottom[self.i_data].diff[...] = top[0].diff
        

