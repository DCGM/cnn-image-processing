'''
Created on May 27, 2016

@author: isvoboda
'''

import cv2
import numpy as np
import os
import logging

class ImageWriter(object):
    """
    Write images.
    """

    def __init__(self, d_path=None, params=None):
        """
        ImageWriter constructor
        Args:
          d_path: directory path where write the image.
        """
        self.d_path = d_path if d_path != None else os.getcwd()
        self.params = params
        self.log = logging.getLogger(__name__ + ".ImageWriter")

    def __call__(self, file_name, img):

        img_name = os.path.join(self.d_path, file_name)
        cv2.imwrite(img_name, img, self.params)


class CoefNpyTxtWriter(object):
    """
    Write numpy nd arrays of jpeg coefs.
    """
    def __init__(self, d_path=None):
        """
        ImageWriter constructor
        Args:
          d_path: directory path where write the coeficients.
        """
        self.d_path = d_path if d_path != None else os.getcwd()
        self.log = logging.getLogger(__name__ + "CoefNpyTxtWriter")

    def __call__(self, file_name, coef):
        coef_file_name = os.path.join(self.d_path, file_name)
        coef_w = coef.transpose([2, 0, 1]).reshape([-1, coef.shape[1]])
        np.savetxt(fname=coef_file_name, X=coef_w, fmt="%3d")
