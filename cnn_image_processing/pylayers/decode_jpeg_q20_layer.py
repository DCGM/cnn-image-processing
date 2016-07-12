'''
Created on Jun 2, 2016

@author: isvoboda
'''

from __future__ import division

import numpy as np
import caffe
import cv2
import logging

MODULE_LOGER = logging.getLogger(__name__)


class PyDecodeJPEGQ20L(caffe.Layer):
    """
    The decode JPEG quant20 layer

    Decodes the dct back into the image space.

    Layer prototxt definition:
    --------------------------
    ToDo
    """

    def setup(self, bottom, top):
        """
        Setup the layer
        """
        self.quant20 = np.asarray([40.,  28.,  25.,  40.,  60., 100., 128.,
                                   153., 30.,  30.,  35.,  48.,  65., 145.,
                                   150., 138., 35.,  33.,  40.,  60., 100.,
                                   143., 173., 140., 35.,  43.,  55.,  73.,
                                   128., 218., 200., 155., 45.,  55.,  93.,
                                   140., 170., 255., 255., 193., 60.,  88.,
                                   138., 160., 203., 255., 255., 230., 123.,
                                   160., 195., 218., 255., 255., 255., 253.,
                                   180., 230., 238., 245., 255., 250., 255.,
                                   248.],
                                  dtype=np.float32)

    def reshape(self, bottom, top):
        """
        Reshape the activation - data blobs
        """
        for i_top in xrange(len(top)):
            shape = np.asarray(bottom[i_top].data.shape)
            shape[2:] = shape[2:] * 8
            shape[1] = 1
            top[i_top].reshape(*shape)

    def forward(self, bottom, top):
        """
        Feed forward
        """
        for i_data in xrange(len(bottom[0].data)):
            coefs = np.transpose(bottom[0].data[i_data], [1, 2, 0])  # [y x z]

            top[0].data[i_data, ...] = self.decode(
                coefs).transpose([2, 0, 1])  # [z y x]

    def backward(self, top, propagate_down, bottom):
        """
        Layer bacpropagation
        """
        for i_diff in xrange(len(top[0].diff)):
            if not propagate_down[i_diff]:
                continue
            top_diff = top[0].diff[i_diff].transpose([1, 2, 0])  # [y x z]
            bottom_data = bottom[0].data[
                i_diff].transpose([1, 2, 0])  # [y x z]
            bottom_diff = bottom[0].diff[i_diff]
            step = 8
            for y_coef in xrange(bottom_data.shape[0]):
                for x_coef in xrange(bottom_data.shape[1]):
                    top_x = step * x_coef
                    top_y = step * y_coef
                    diff_block = top_diff[
                        top_y:top_y + step, top_x:top_x + step]
                    # 1. Compute derivative of f: idct(x) / d_bottom
                    d_bottom_d_top = cv2.dct(diff_block)
                    d_bottom_d_top = d_bottom_d_top.reshape(64) * self.quant20
                    bottom_diff[:, y_coef, x_coef] = d_bottom_d_top

    def decode(self, coefs):
        """
        Decode the coefs back to the pixels
        Provide clipping from -127 to 128 - data are shifted by 128.
        """
        img = np.zeros([dim * 8 for dim in coefs.shape[0:2]], dtype=np.double)
        step = 8
        for y_coef in xrange(coefs.shape[0]):
            for x_coef in xrange(coefs.shape[1]):
                in_x = step * x_coef
                in_y = step * y_coef
                img[in_y:in_y + step, in_x:in_x + step] = \
                    cv2.idct(
                        (coefs[y_coef, x_coef] * self.quant20).reshape([8, 8]))

        img = np.clip(img, -127, 128)
        return np.expand_dims(img, axis=2)
