'''
Created on May 27, 2016

@author: isvoboda
'''

from __future__ import print_function
from __future__ import division

import caffe
import logging
import numpy as np
from ..utils import RoundBuffer

class PyPSNRL(caffe.Layer):
    """
    Compute a PSNR of two inputs and PSNR with IPSNR of 3 inputs
    Args:
        max: float
            The maximum value in the input data used to compute the PSNR.
            Default is 255
    history_size: uint
         Size of history a floating avarge is computed from.
    """
    def setup(self, bottom, top):
        self.log = logging.getLogger(__name__ + ".PyPSNRL")
        if len(bottom) < 2 or len(bottom) > 3:
            raise Exception("Need two inputs at least or 3 at most.")
        self.dict_param = dict((key.strip(), val.strip()) for key, val in
                               (item.split(':') for item in
                                self.param_str.split(',')))
        if 'max' in self.dict_param:
            self.max = float(self.dict_param['max'])
        else:
            self.max = 255
        if 'history_size' in self.dict_param:
            self.history_size = np.uint(self.dict_param['history_size'])
        else:
            self.history_size = 50
            
        self.psnr_buffer = RoundBuffer(max_size=self.history_size)
       
    def reshape(self, bottom, top):
        if len(top) > len(bottom):
            raise Exception("Layer produce more outputs then has its inputs.")
        
        for i_input in xrange(len(top)):
            top[i_input].reshape(*bottom[i_input].data.shape)
    
    def forward(self, bottom, top):
        psnr = self.psnr(bottom)
        msg = "\n".join('PSNR {}: {}'.format(*val) for val in enumerate(psnr))
        self.log.info(msg)
        if len(psnr) == 2:
            self.log("iPSNR: {}".format(psnr[0] - psnr[1]))
    
    def backward(self, top, propagate_down, bottom):
        for i_diff in xrange(len(top)):
            bottom[i_diff].diff[...] = top[i_diff].diff
    
    def psnr(self, bottom):
        results = []
        for i_input in xrange(len(bottom)-1):
            diff = bottom[-1].data - bottom[i_input].data
            ssd = (diff**2).sum()
            mse = ssd / float(diff.size)
            if mse <= 0:
                results.append(np.nan)
            else:
                psnr = 10 * np.log10(self.max**2 / mse)
                results.append(psnr)
        return results