'''
Created on May 27, 2016

@author: isvoboda
'''

import logging
import os.path
import cv2
import numpy as np


class ImageReader(object):

    """
    Reads several types of images via OpenCV.
    Always returns the float 3dim numpy array.
    """

    def __init__(self, grayscale=False):
        '''
        Initialize the image reader

        args:
            grayscale: Boolean
                True: read the image as is
                False (default): Read grayscale image with pne channel [Y,X,1]
        '''
        self.grayscale = grayscale
        self.load_flag = cv2.IMREAD_GRAYSCALE if grayscale\
            else cv2.IMREAD_UNCHANGED
        self.log = logging.getLogger(".".join([__name__, type(self).__name__]))

    def read(self, targs):
        """
        Loads and returns the image of path.
        """
        img = None
        try:
            packet = targs.packet
            path = packet.path
            packet.data = None
            img = cv2.imread(path, self.load_flag).astype(np.float32)

            if self.grayscale is True:
                img = img.reshape(img.shape[0], img.shape[1], 1)

            packet.data = img
            return targs

        except cv2.error:
            self.log.exception("cv2.error")
            return targs
        except IOError:
            self.log.exception("Error reading %s", packet.path)
            return targs
        except AttributeError:
            self.log.exception("No path defined")
            return targs

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
        self.log = logging.getLogger(".".join([__name__, type(self).__name__]))

    def read(self, targs):
        """
        Loads and returns the cropped image size divideable by 8.
        """
        try:
            targs = super(ImageX8Reader, self).read(targs)
            packet = targs.packet
            packet.data = self.crop(packet.data)
            return targs

        except TypeError:
            self.log.exception("Failed")
            return targs

    def crop(self, img):
        """
        Crop the img to be 8x divisible.
        """
        crop_shape = [i - (i % 8) for i in img.shape[0:2]]
        return img[0:crop_shape[0], 0:crop_shape[1]]


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
            txtpath = os.path.splitext(path)[0] + ".txt"
            npzpath = os.path.splitext(path)[0] + ".npz"
            if os.path.isfile(npzpath):
                data_array = np.load(file=npzpath)
                data_array = data_array[data_array.keys()[0]]
                data_array = data_array.astype(np.float32)
            elif os.path.isfile(txtpath):
                data_array = np.loadtxt(fname=txtpath, dtype=np.float32)
            else:
                raise Exception('none of those files exist')
            if data_array is None:
                self.log.error("Could not read data: %r", path)
            data_array = data_array.reshape([self.n_channels, -1,
                                             data_array.shape[1]])
            data_array = data_array.transpose([1, 2, 0])  # [y,x,z]
        except IOError:
            self.log.exception("IOError")

        packet['data'] = data_array
        return packet

    def __call__(self, packet):
        """
        Return the JPEG coefs stored in the txt file in path.
        """
        return self.read(packet)
