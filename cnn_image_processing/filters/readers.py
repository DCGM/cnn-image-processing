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

    def __init__(self, grayscale=bool):
        self.load_flag = cv2.IMREAD_GRAYSCALE if grayscale\
            else cv2.IMREAD_COLOR
        self.log = logging.getLogger(__name__ + ".ImageReader")

    def read(self, packet):
        """
        Loads and returns the image of path.
        """
        img = None
        try:
            path = packet['path']
            img = cv2.imread(path, self.load_flag)
            if img is None:
                self.log.error("Could not read image: %r", path)
                return None
            img = img.astype(np.float32)
        except cv2.error:
            self.log.exception("cv2.error")
        except IOError as ioe:
            print "I/O error({0}): {1}".format(ioe.errno, ioe.strerror)
        except:
            self.log.exception("UNKNOWN")
        if len(img.shape) == 2:
            img = img.reshape(img.shape[0], img.shape[1], 1)
        packet['data'] = img
        return packet

    def __call__(self, path):
        """
        Returns the image
        """
        return self.read(path)

class CameraReader(object):
    """
    Read images from camera
    """

    def __init__(self, grayscale=bool, crop_size=250):
        self.load_flag = cv2.IMREAD_GRAYSCALE if grayscale\
            else cv2.IMREAD_COLOR
        self.capture = cv2.VideoCapture(0)
        self.cropSize = crop_size
        self.log = logging.getLogger(__name__ + ".CameraReader")

    def read(self, packet):
        """
        Loads and returns the image of path.
        """
        ret, frame = self.capture.read()
        if not ret:
            self.log.error("Could not read image fom camera.")
            exit(-1)
        img = frame

        b = ((img.shape[0]-self.cropSize)/2, (img.shape[1]-self.cropSize)/2)
        img = img[b[0]:b[0]+self.cropSize, b[1]:b[1]+self.cropSize,:]
        #cv2.imshow('in', cv2.resize(img, (0,0), fx=3, fy=3))


        img[:,:,:] = np.expand_dims(cv2.cvtColor(src=img, code=cv2.COLOR_BGR2GRAY), axis=2)
        img = img.astype(np.float32)
        #img /= img.max()
        #img -= img.min() + 0.1
        img /= 255
        img -= 0.5
        #cv2.waitKey(5)
        img = img.astype(np.float32)
        if len(img.shape) == 2:
            img = img.reshape(img.shape[0], img.shape[1], 1)
        packet['data'] = img
        return packet

    def __call__(self, path):
        """
        Returns the image
        """
        return self.read(path)

class TupleReader(object):

    """
    Reads tuple of float values separated by ',' - e.g. 1.2,5,3.4
    """

    def __init__(self):
        self.log = logging.getLogger(__name__ + ".TupleReader")

    def __call__(self, packet):
        """
        Converts tuple string to numpy array
        """
        try:
            packet['data'] = [float(x) for x in packet['path'].split(',')]
            packet['data'] = np.asarray(packet['data']).astype(np.float32).reshape(1,1,-1)
        except:
            self.log.exception("Failed to parse float tuple '{}'".format(packet['path']))
        return packet


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
