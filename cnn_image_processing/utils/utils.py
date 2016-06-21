'''
Created on Apr 11, 2016

@author: isvoboda, ihradis

'''
from __future__ import print_function
from __future__ import division

import numpy as np
import cv2

def code_dct(img):
    """
    Code the img into the coefs.
    Image is the y * x * 1 NDarray and coefs are y/8 * x/8 * 64 NDarray.
    Where 1*1*64 is one block of coefficients
    equivalent to the 8*8*1 image pixels.
    
    Args:
        img: Numpy NDArray [y,x,z]
    """
    coef_shape = [ dim // 8 for dim in img.shape[0:2]]
    coef_shape.append(64)
    coefs = np.zeros( coef_shape, dtype=np.double)
    step = 8
    for y_coef in xrange(coefs.shape[0]):
        for x_coef in xrange(coefs.shape[1]):
            in_x = step*x_coef
            in_y = step*y_coef 
            coefs[y_coef, x_coef] = \
                cv2.dct(img[in_y:in_y+step, in_x:in_x+step] - 128).reshape(64)

    coefs /= 1024
    return coefs

def decode_dct(coefs):
    """
    Decodes the dct coefs into image.
    Coefs is the x * y * 64 NDarray where 1*1*64 is one block of coefficients
    equivalent to the 8*8*1 image pixels.
    
    Args:
        coefs: Numpy NDArray [y,x,z]
    """
#     coefs_shape = coefs.transpose()
    img = np.zeros([ dim * 8 for dim in coefs.shape[0:2]], dtype=np.double)
    step = 8
    for y_coef in xrange(coefs.shape[0]):
        for x_coef in xrange(coefs.shape[1]):
            in_x = step*x_coef
            in_y = step*y_coef 
            img[in_y:in_y+step, in_x:in_x+step] = \
                cv2.idct(coefs[y_coef, x_coef].reshape([8,8]) * 1024)

    img += 128
    return np.expand_dims(img, axis=2)
