'''
Created on Apr 11, 2016

@author: isvoboda, ihradis

'''
from __future__ import print_function

# import threading
import cv2

def decode_dct(coefs):
    """
    Compute the inverse dct of the coefs.
    Coefs is the x * y * 64 NDarray where 1*1*64 is one block of coefficients
    equivalent to the 8*8 image pixels.
    """
#     coefs_shape = coefs.transpose()
    img = np.zeros([ dim * 8 for dim in coefs.shape[0:2]], dtype=np.double)
    step = 8
    for y_coef in xrange(coefs.shape[0]):
        for x_coef in xrange(coefs.shape[1]):
            in_x = step*x_coef
            in_y = step*y_coef 
            img[in_y:in_y+step, in_x:in_x+step] = \
                cv2.idct(coefs[y_coef, x_coef].reshape([8,8])) * 1024

    img += 128
    return img
