'''
Created on May 27, 2016

@author: isvoboda
'''

import cv2
import numpy as np
import logging
import sys

class ImageReader(object):
    """
    Reads several types of images via OpenCV.
    Always returns the float 3dim numpy array.
    """
    def __init__(self):
        self.log = logging.getLogger(__name__ + ".ImageReader")

    def read(self, path):
        """
        Loads and returns the image on path.
        """
        img = None
        try:
            img = cv2.imread(path, -1).astype(np.float)
            if img is None:
                self.log.error("Could not read image: {}".format(path))
                return None
        except cv2.error as err:
            self.log.error("cv2.error: {}".format(str(err)))
        except:
            self.log.error("UNKNOWN: {}".format(sys.exc_info()[0]))
        if len(img.shape) == 2:
            img = img.reshape(img.shape[0],img.shape[1],1)
        return img
    
    def __call__(self, path):
        """
        Returns the image
        """
        return self.read(path)

class ImageX8Reader(ImageReader):
    """
    Reads several types of images via OpenCV.
    Crop the images to have the size divideable by 8.
    Pivot for cropping is the image left up corner.
    """
    log = logging.getLogger(__name__ + ".ImageX8Reader")

    @classmethod
    def read(cls, path):
        """
        Loads and returns the cropped image size divideable by 8.
        """
        img = super(ImageX8Reader, cls).read(path)
        if img != None:
            return ImageX8Reader.crop(img)
        else:
            return img

    @classmethod
    def crop(cls, img):
        """
        Crop the img to be 8x divisible.
        """
        crop_c = [i - (i % 8) for i in img.shape[0:2]]
        return img[0:crop_c[0], 0:crop_c[1], ...]

class CoefNpyTxtReader(object):
    """
    Reads the jpeg-coefs in numpy array txt file format.
    Always returns float Xdim numpy array.
    """
    def __init__(self, n_channels=64):
        self.n_channels = n_channels
        self.log = logging.getLogger(__name__ + ".CoefNpyTxtReader")

    def read(self, path):
        """
        Reads the numpy txt array file and returns the numpy array.
        """
        data_array = None
        try:
            data_array = np.loadtxt(fname=path, dtype=np.float)
            if data_array is None:
                self.log.error("Could not read data: {}".format(path))
            data_array = data_array.reshape([self.n_channels, -1,
                                             data_array.shape[1]])
            data_array = data_array.transpose([1, 2, 0])  # [y,x,z]
        except IOError as ioe:
            self.log.error("IOError: {}".format(str(ioe)))

        return data_array
    
    def __call__(self, path):
        """
        Return the JPEG coefs stored in the txt file in path.
        """
        return self.read(path)
