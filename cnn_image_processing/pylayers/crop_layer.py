'''
Created on May 27, 2016

@author: isvoboda
'''

from __future__ import print_function
from __future__ import division

import numpy as np
import caffe


class PyCropL(caffe.Layer):
    """
    Crop the center of bottom[0].data according the bottom[1].data
    """

    def setup(self, bottom, top):
        """
        Setup the layer
        """
        if len(bottom) != 2:
            raise Exception("PyCropL has to have 2 inputs.")
        self.i_data = 0
        self.i_label = 1
        # Compute the border
        label_shape = np.asarray(bottom[self.i_label].data.shape[2:])
        data_shape = np.asarray(bottom[self.i_data].data.shape[2:])
        self.borders = data_shape - label_shape
        if not np.all(self.borders >= 0):
            raise Exception("Bottom input 0 is smaller then the"
                            " crop refference 1.")
        self.borders //= 2

    def reshape(self, bottom, top):
        """
        Reshape the activation - data blobs
        """
        top[0].reshape(*bottom[self.i_label].data.shape)

    def forward(self, bottom, top):
        """
        Feed forward
        """
        (i_crop_x, i_crop_y) = self.borders
        len_x = bottom[self.i_label].data.shape[2]
        len_y = bottom[self.i_label].data.shape[3]
        data = bottom[self.i_data].data
        crop_data = data[:, :, i_crop_x:i_crop_x + len_x,
                         i_crop_y:i_crop_y + len_y]
        top[0].data[...] = crop_data

    def backward(self, top, propagate_down, bottom):
        """
        Layer bacpropagation
        """
        if propagate_down[self.i_data]:
            pad_params = ((0, 0), (0, 0), self.borders, self.borders)
            bottom[self.i_data].diff[...] = np.pad(top[0].diff, pad_params,
                                                   mode='constant',
                                                   constant_values=(0, 0))
