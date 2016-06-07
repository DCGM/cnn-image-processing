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
            Default is 50
        print_step: uint
            Ith iteration to print the PSNR
            Default is 50
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
            self.history_size = int(self.dict_param['history_size'])
        else:
            self.history_size = 50
        if 'print_step' in self.dict_param:
            self.print_step = int(self.dict_param['print_step'])
        else:
            self.print_step = 50
        
        self.iterations = 0
        self.psnr_buffers = [RoundBuffer(max_size=self.history_size)
                             for _ in xrange(len(bottom)-1)]
       
    def reshape(self, bottom, top):
        if len(top) > len(bottom):
            raise Exception("Layer produce more outputs then has its inputs.")
        
        for i_input in xrange(len(top)):
            top[i_input].reshape(*bottom[i_input].data.shape)
    
    def forward(self, bottom, top):
        l_psnr = self.psnr(bottom)
        for i_val, val in enumerate(l_psnr):
            self.psnr_buffers[i_val].append_round(val)
        
        if self.iterations % self.print_step == 0: 
            avg_psnr = [sum(val)/val.size for val in self.psnr_buffers]
                        
            avg_msg = " PSNR average of {} samples".format(self.psnr_buffers[0].size)
            self.log.info(avg_msg)
            
            msg = " ".join(' PSNR bottom[{}]: {}'
                        .format(*val) for val in enumerate(avg_psnr))
            
            self.log.info(msg)
            if len(l_psnr) == 2:
                self.log.info(" iPSNR: {}".format(avg_psnr[0] - avg_psnr[1]))     
            
        for i_data in xrange(len(top)):
            top[i_data].data[...] = bottom[i_data].data
        
        self.iterations += 1 
    
    def backward(self, top, propagate_down, bottom):
        for i_diff in xrange(len(top)):
            bottom[i_diff].diff[...] = top[i_diff].diff
    
    def psnr(self, bottom):
        results = []
        for i_input in xrange(len(bottom)-1):
            diff = (bottom[-1].data - bottom[i_input].data).astype(np.float64)
            ssd = (diff**2).sum()
            mse = ssd / float(diff.size)
            if mse == 0:
                results.append(np.nan)
            else:
                psnr = 10 * np.log10(self.max**2 / mse)
                results.append(psnr)
        return results