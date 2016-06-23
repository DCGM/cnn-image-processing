'''
Created on May 27, 2016

@author: isvoboda
'''

import cv2
import numpy as np
import logging
import sys
import os.path

class ImageReader(object):
    """
    Reads several types of images via OpenCV.
    Always returns the float 3dim numpy array.
    """
    def __init__(self, grayscale=bool):
        self.load_flag = cv2.IMREAD_GRAYSCALE if grayscale\
                                              else cv2.IMREAD_UNCHANGED 
        self.log = logging.getLogger(__name__ + ".ImageReader")

    def read(self, packet):
        """
        Loads and returns the image of path.
        """
        img = None
        try:
            path = packet['path']
            img = cv2.imread(path, self.load_flag).astype(np.float32)
            if img is None:
                self.log.error("Could not read image: {}".format(path))
                return None
        except cv2.error as err:
            self.log.error("cv2.error: {}".format(str(err)))
        except:
            self.log.error("UNKNOWN: {}".format(sys.exc_info()[0]))
        if len(img.shape) == 2:
            img = img.reshape(img.shape[0],img.shape[1],1)
        packet['data'] = img
        return packet
    
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
    def __init__(self, grayscale=bool):
        super(ImageX8Reader, self).__init__(grayscale)
        self.log = logging.getLogger(__name__ + ".ImageX8Reader")

    def read(self, packet):
        """
        Loads and returns the cropped image size divideable by 8.
        """
        packet = super(ImageX8Reader, self).read(packet)
        if packet != None:
            packet['data'] = self.crop(packet['data'])
            return packet
        else:
            return None

    def crop(self, img):
        """
        Crop the img to be 8x divisible.
        """
        crop_shape = [i - (i % 8) for i in img.shape[0:2]]
        return img[0:crop_shape[0], 0:crop_shape[1], ...]

class CoefNpyTxtReader(object):
    """
    Reads the jpeg-coefs in numpy array txt file format.
    Always returns float Xdim numpy array.
    """
    def __init__(self, n_channels=64):
        self.n_channels = n_channels
        self.log = logging.getLogger(__name__ + ".CoefNpyTxtReader")

    def read(self, packet):
        """
        Reads the numpy txt array file and returns the numpy array.
        """
        path = packet['path']
        data_array = None
        try:
            txtpath = os.path.splitext(path)[0] + ".txt";
            npzpath = os.path.splitext(path)[0] + ".npz";
            if   os.path.isfile(npzpath):
                data_array = np.load(file=npzpath)
                data_array = data_array[ data_array.keys()[0] ]
                data_array = data_array.astype(np.float32)
            elif os.path.isfile(txtpath):
                data_array = np.loadtxt(fname=txtpath, dtype=np.float32)
            else:
                raise Exception('none of those files exist');
            if data_array is None:
                self.log.error("Could not read data: {}".format(path))
            data_array = data_array.reshape([self.n_channels, -1,
                                             data_array.shape[1]])
            data_array = data_array.transpose([1, 2, 0])  # [y,x,z]
        except IOError as ioe:
            self.log.error("IOError: {}".format(str(ioe)))

        packet['data'] = data_array
        return packet
    
    def __call__(self, packet):
        """
        Return the JPEG coefs stored in the txt file in path.
        """
        return self.read(packet)
