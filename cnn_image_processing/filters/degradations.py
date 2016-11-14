from __future__ import division
from __future__ import print_function

import logging
import cv2
import numpy as np

from ..utilities import parameter, Configurable, ContinuePipeline


class JPEG(Configurable):
    '''
    JPEG encode and decode the input data with specified quality

    Example:
    JPEG: {quality: 20}
    '''
    def addParams(self):
        self.params.append(parameter(
            'quality', required=False, default=20, parser=int,
            help='JPEG qualilty. 100 for maximum quality; 1 for minimum quality.'))
        self.params.append(parameter(
            'quality_low', required=False, default=None, parser=int,
            help='Minimum JPEG qualilty.'))
        self.params.append(parameter(
            'quality_high', required=False, default=None, parser=int,
            help='Maximum JPEG qualilty.'))

    def __init__(self, config):
        Configurable.__init__(self)
        self.log = logging.getLogger(__name__ + "." + type(self).__name__)
        self.addParams()
        self.parseParams(config)

        if self.quality_low and self.quality_high:
            self.quality = None
        else:
            self.quality_low = self.quality
            self.quality_high = self.quality + 1

        if self.quality_low >= self.quality_high:
            msg = 'quality_low ({}) has to be lower then quality_high ({}). Config Line {}.'.format(self.quality_low, self.quality_high, self.line)
            self.log.error(msg)
            raise ValueError(msg)

    def __call__(self, packet, previous):
        '''
        Compress the input packet's data with the jpeg encoder
        '''
        quality = np.random.randint(self.quality_low, self.quality_high)
        res, img_str = cv2.imencode('.jpeg', packet['data'],
                                    [cv2.IMWRITE_JPEG_QUALITY, quality])
        if not res:
            msg = ' Failed to JPEG encode data. Config Line {}'.format(
                self.line)
            self.log.error(msg)
            raise ContinuePipeline

        img = cv2.imdecode(np.asarray(bytearray(img_str), dtype=np.uint8),
                           cv2.IMREAD_UNCHANGED)
        packet['data'] = img.reshape(img.shape[0], img.shape[1], -1)
        return [packet]


class ColorBalance(Configurable):
    def __init__(self, color_sdev):
        self.color_sdev = color_sdev

    def __call__(self, packet):
        img = packet['data']
        colorCoeff = [2**(np.random.standard_normal() * self.color_sdev) for x in range(img.shape[2])]
        for i, coef in enumerate(colorCoeff):
            img[:, :, i] *= coef
        packet['data'] = img
        return packet


class Noise(object):
    def __init__(self, min_noise, max_noise):
        self.min_noise = min_noise
        self.max_noise = max_noise

    def __call__(self, packet):
        noiseSdev = np.random.uniform(self.min_noise, self.max_noise)
        packet['data'] = packet['data'] + np.random.randn(*packet['data'].shape) * noiseSdev
        return packet


class ClipValues(object):
    def __init__(self, minVal=0, maxVal=255):
        self.minVal = minVal
        self.maxVal = maxVal

    def __call__(self, packet):
        packet['data'] = np.maximum(np.minimum(packet['data'], self.maxVal), self.minVal)
        return packet


class GammaCorrection(object):
    """
    Runs the image throug gamma correction:
    im = im**gamma / (255**gamma) * 255,
    where
    gamma = 2**normal(gamma_sdev)
    """
    def __init__(self, gamma_sdev):
        self.gamma_sdev = gamma_sdev

    def __call__(self, packet):
        gamma = 2**(np.random.standard_normal() * self.gamma_sdev)
        packet['data'] = (packet['data'].copy().astype(np.float32)**gamma) / (255**gamma) * 255
        return packet

class ReduceContrast(object):
    """
    Can reduce contrast by randomly shifting zero intesity up
    (up to min_intensity) and shifting highest intensity down
    (at most to max_intensity).
    """
    def __init__(self, min_intensity, max_intensity):
        self.min_intensity = min_intensity
        self.max_intensity = max_intensity

    def __call__(self, packet):
        minVal = np.random.uniform(0, self.min_intensity)
        maxVal = np.random.uniform(self.max_intensity, 255)
        packet['data'] = packet['data'] / 255.0 * (maxVal - minVal) + minVal
        return packet
