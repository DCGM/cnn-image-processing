'''
Created on Jun 23, 2016

@author: isvoboda
'''
from __future__ import division

import numpy as np
import caffe
import logging

module_logger = logging.getLogger(__name__)

class PyDeBlockJPEG(caffe.Layer):
    
    def setup(self, bottom, top):
        self.log = logging.getLogger(__name__)
        assert(len(top) == len(bottom))
            
    def reshape(self, bottom, top):        
        for i_top in xrange(len(top)):
            shape = np.asarray(bottom[i_top].data.shape)
            shape[2:] = shape[2:]*8
            shape[1] = 1
            top[i_top].reshape(*shape)

    def forward(self, bottom, top):
        for i_bottom in xrange(len(bottom)):
            for i_data in xrange(len(bottom[i_bottom].data)):
                data = np.transpose(bottom[i_bottom].data[i_data], [1,2,0]) #[y x z]
                deblock_data = self.deblock(data).transpose([2,0,1]) #[z y x]
                top[i_bottom].data[i_data, ...] = deblock_data 
    
    def backward(self, top, propagate_down, bottom):           
        for i_top in xrange(len(top)):
            if propagate_down[i_top] != True:
                continue
            for i_diff in xrange(len(top[0].diff)):
                top_diff = top[i_top].diff[i_diff].transpose([1,2,0]) #[y x z]
                bottom_diff = bottom[i_top].diff[i_diff]
                step = 8
                for y_id in xrange(bottom_diff.shape[1]):
                    for x_id in xrange(bottom_diff.shape[2]):
                        in_y = step*y_id
                        in_x = step*x_id
                        diff_data = top_diff[in_y:in_y+step,
                                             in_x:in_x+step].reshape(64)
                        bottom_diff[..., y_id, x_id] = diff_data 
    
    def deblock(self, data):
        img = np.zeros([ dim * 8 for dim in data.shape[0:2]], dtype=np.double)
        step = 8
        for y_coef in xrange(data.shape[0]):
            for x_coef in xrange(data.shape[1]):
                in_x = step*x_coef
                in_y = step*y_coef 
                img[in_y:in_y+step, in_x:in_x+step] = data[y_coef, x_coef]\
                .reshape([8,8])
     
        return np.expand_dims(img, axis=2)
