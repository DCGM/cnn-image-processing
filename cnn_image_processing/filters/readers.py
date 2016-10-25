from __future__ import print_function

import logging
import os.path
import cv2
import numpy as np

from ..utilities import parameter, Configurable, ContinuePipeline, TerminatePipeline

class ListFileReader(Configurable):

    """
    Reads a file and parses lines into white-space separated items.
    It should be used as the first filter in image reading pipelines.

    Example:
    ListFileReader: {file_name: "file.txt", loop: True}
    """

    def addParams(self):
        self.params.append(parameter(
            'file_name', required=True, parser=str,
            help='File name string'))
        self.params.append(parameter(
            'loop', default=True, parser=bool,
            help='Should the reader loop over the file infinitely?'))

    def __init__(self, config):
        Configurable.__init__(self)
        self.log = logging.getLogger(__name__ + "." + type(self).__name__)
        self.addParams()
        self.parseParams(config)

        try:
            self.file = open(self.file_name, 'r')
            line = self.file.readline()
            if len(line) == 0:
                self.log.error("File '%s' is empty", self.file_name)
                raise IOError("File is empty")
            self.file.seek(0)
        except IOError as ex:
            self.log.error("Failed to open file '%s'", self.file_name)
            self.log.error(ex)
            raise ex

    def __call__(self, packet, previous):
        line = self.file.readline()
        if len(line) == 0:
            if self.loop:
                self.file.seek(0)
                raise ContinuePipeline
            else:
                self.log.info('Finished reading file ')
                raise TerminatePipeline

        packets = [{'data': data.strip()} for data in line.split()]
        return packets


class ImageReader(Configurable):

    """
    Reads several types of images via OpenCV.
    Always returns the float 3dim numpy array.

    Example:
    ImageReader: {grayscale: True}
    """

    def addParams(self):
        self.params.append(parameter(
            'grayscale', required=False, default=False, parser=bool,
            help='All images will be converted to grayscale - reading RGB otherwise.'))

    def __init__(self, config):
        Configurable.__init__(self)
        self.log = logging.getLogger(__name__ + "." + type(self).__name__)
        self.addParams()
        self.parseParams(config)

        self.load_flag = cv2.IMREAD_GRAYSCALE if self.grayscale\
            else cv2.IMREAD_COLOR

    def __call__(self, packet, previous):
        """
        Loads and returns the image from packet['data'].
        """
        img = None
        try:
            packet['path'] = packet['data']
            path = packet['data']
            img = cv2.imread(path, self.load_flag).astype(np.float32)
            if len(img.shape) == 2:
                img = img.reshape(img.shape[0], img.shape[1], 1)
        except (cv2.error, Exception):
            self.log.warning('Unable to read file "{}"'.format(path),
                             exc_info=True)
            raise ContinuePipeline

        packet['data'] = img
        return [packet]


class TupleReader(Configurable):

    """
    Reads tuple of float values separated by ',' - e.g. 1.2,5,3.4

    Example:
    TupleReader: {}
    """

    def __init__(self, config):
        Configurable.__init__(self)
        self.log = logging.getLogger(__name__ + "." + type(self).__name__)
        self.addParams()
        self.parseParams(config)

    def __call__(self, packet, previous):
        """
        Converts tuple string to numpy array
        """
        try:
            packet['data'] = [float(x) for x in packet['data'].split(',')]
            packet['data'] = np.asarray(packet['data']).astype(np.float32).reshape(1, 1, -1)
        except Exception:
            self.log.exception("Failed to parse float tuple '{}'".format(packet['data']))
            raise ContinuePipeline
        return [packet]


class ImageX8Reader(Configurable):
    """
    Crops image to have the size divisible by 8.
    Pivot for cropping is the image left up corner.

    Example:
    ImageX8Reader: {}
    """

    def __init__(self, config):
        Configurable.__init__(self)
        self.log = logging.getLogger(__name__ + "." + type(self).__name__)
        self.addParams()
        self.parseParams(config)

    def __call__(self, packet, previous):
        img = packet['data']
        crop_shape = [i - (i % 8) for i in img.shape[0:2]]
        packet['data'] = img[0:crop_shape[0], 0:crop_shape[1], ...]
        return [packet]


class CoefNpyTxtReader(Configurable):
    """
    Reads the jpeg-coefs in numpy array txt file format.
    Always returns float Xdim numpy array.

    Example:
    CoefNpyTxtReader: {n_channels=64}
    """

    def addParams(self):
        self.params.append(parameter(
            'n_channels', required=False, default=64, parser=int,
            help='Number of channels.'))

    def __init__(self, config):
        Configurable.__init__(self)
        self.log = logging.getLogger(__name__ + "." + type(self).__name__)
        self.addParams()
        self.parseParams(config)

    def __call__(self, packet, previous):
        """
        Reads the numpy txt array file and returns the numpy array.
        """
        packet['path'] = packet['data']
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
                self.log.waring("File does not exist with any acceptable extension '{}' or '{}'.".format(txtpath, npzpath))
                raise ContinuePipeline
            data_array = data_array.reshape([self.n_channels, -1,
                                             data_array.shape[1]])
            data_array = data_array.transpose([1, 2, 0])  # [y,x,z]
        except IOError:
            self.log.exception("Uable to read file '{}'".format(packet['data']))
            raise ContinuePipeline

        packet['data'] = data_array
        return [packet]
