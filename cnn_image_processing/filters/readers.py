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
            self.log.exception("OpenCv")
            raise
        except IOError:
            self.log.exception("Error reading %r", packet.path)
            raise
        except AttributeError:
            self.log.exception("Attribute error")
            raise

    def __call__(self, targs):
        """
        Returns the image
        """
        return self.read(targs)


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
            raise

    def crop(self, img):
        """
        Crop the img to be 8x divisible.
        """
        crop_shape = [i - (i % 8) for i in img.shape[0:2]]
        return img[0:crop_shape[0], 0:crop_shape[1]]


class CoefNpyReader(object):

    """
    Reads the jpeg-coefs in numpy array txt file format.
    Always returns float Xdim numpy array.
    """

    def __init__(self, n_channels=64):
        '''
        Initilize the CoefNpyReader
        '''
        self.n_channels = n_channels
        self.log = logging.getLogger(".".join([__name__, type(self).__name__]))

    def read(self, targs):
        """
        Reads the numpy txt/binary array file
        """
        try:
            packet = targs.packet
            path = packet.path
            data_array = None
            txtpath = os.path.splitext(path)[0] + ".txt"
            npzpath = os.path.splitext(path)[0] + ".npz"
            if os.path.isfile(npzpath):
                data_array = np.load(file=npzpath)
                data_array = data_array[data_array.keys()[0]]
                data_array = data_array.astype(np.float32)
            elif os.path.isfile(txtpath):
                data_array = np.loadtxt(fname=txtpath, dtype=np.float32)
            else:
                raise IOError('None of {}, {} exist'.format(
                    npzpath, txtpath))

            data_array = data_array.reshape([self.n_channels, -1,
                                             data_array.shape[1]])
            packet.data = data_array.transpose([1, 2, 0])  # [y,x,z]

            return targs

        except IOError:
            self.log.exception("IOError")
            raise

        except Exception:
            self.log.exception("Exception")
            raise

    def __call__(self, targs):
        return self.read(targs)
