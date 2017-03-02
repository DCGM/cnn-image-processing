from __future__ import division
from __future__ import print_function

import logging
import cv2
import numpy as np
import functools
import itertools
from pandas import ewma

from ..utilities import parameter, Configurable, ContinuePipeline


class MotionBlur(Configurable):
    '''
    Blurs images with linear motion kernels.
    The kernels are sampled from a range of directions and lenths.
    Directions are defined in degrees - 0-180 is the larges possible range.

    Example:
    MotionBlur: {dir_range: (0, 180), length_range: (0,8)}
    '''
    def addParams(self):
        self.params.append(parameter(
            'dir_range',
            required=False,
            parser=lambda x: [float(i) for i in x],
            default=(0, 180),
            help='Pair of values indicatig range of directions.'))
        self.params.append(parameter(
            'length_range',
            parser=lambda x: [float(i) for i in x],
            default=(0, 8),
            help='Pair of values indicatig range of lengths in pixels.'))
        self.params.append(parameter(
            'rng_seed', required=False, default=None,
            parser=lambda x: max(int(x), 1),
            help='Size of the buffer'))
        self.params.append(parameter(
            'motion_packet', required=False, default=False,
            parser=bool,
            help='If true, will add new packet with single value encoding the motion filter (direction and length).'))
        self.params.append(parameter(
            'length_bins', required=False, default=6,
            parser=int,
            help='Number of bins encoding motion length.'))
        self.params.append(parameter(
            'dir_bins', required=False, default=12,
            parser=int,
            help='Number of bins encoding motion orientations.'))

    def verifyParams(self):
        if len(self.dir_range) != 2 or len(self.length_range) != 2:
            msg = ' dir_range and length_range have to be pairs of values. Config Line {}'.format(
                self.line)
            self.log.error(msg)
            raise AttributeError(msg)

        self.length_range = sorted([max(0, x) for x in self.length_range])
        self.dir_range = sorted(self.dir_range)

    def __init__(self, config):
        Configurable.__init__(self)
        self.log = logging.getLogger(__name__ + "." + type(self).__name__)
        self.addParams()
        self.parseParams(config)
        self.verifyParams()
        self.log.info('dir_range is {} and length_range is {}'.format(
            self.dir_range, self.length_range))
        if self.rng_seed:
            self.rng = np.random.RandomState(self.rng_seed)
        else:
            self.rng = np.random.RandomState()

        step = (self.length_range[1] - self.length_range[0]) / self.length_bins
        length_bins = [
            (self.length_range[0] + x * step, self.length_range[0] +x * step + step)
            for x in range(self.length_bins)]
        step = (self.dir_range[1] - self.dir_range[0]) / self.dir_bins
        dir_bins = [
            (self.dir_range[0] + x * step, self.dir_range[0] + x * step + step)
            for x in range(self.dir_bins)]
        self.bins = list(itertools.product(length_bins, dir_bins))
        self.next_bins = []
        self.log.info(' Have {} blur bins.'.format(len(self.bins)))
        self.log.info(' The bins are: {}'.format(self.bins))

    def __call__(self, packet, previous):
        if not self.next_bins:
            self.next_bins = list(np.random.permutation(self.bins))
        length_range, dir_range = self.next_bins.pop()

        sampled_slope_deg = self.rng.uniform(
            low=dir_range[0], high=dir_range[1])
        sampled_length = self.rng.uniform(
            low=length_range[0], high=length_range[1])

        psf = self.generateMotionBlurPSF(sampled_slope_deg, sampled_length)

        packet['motion_blur_slope'] = sampled_slope_deg
        packet['motion_blur_length'] = sampled_length
        if 'psf' in packet:
            border0 = np.ceil(packet['psf'].size[0] / 2)
            border1 = np.ceil(packet['psf'].size[0] / 2)
            psf = cv2.copyMakeBorder(
                psf, border0, border0, border1, border1,
                cv2.BORDER_CONSTANT, value=0)
            psf = cv2.filter2D(
                psf, -1, packet['psf'], borderType=cv2.BORDER_REPLICATE)
            psf = psf
        packet['psf'] = psf

        packet['data'] = cv2.filter2D(
            packet['data'].astype(np.float32), cv2.CV_32F, psf,
            borderType=cv2.BORDER_REPLICATE)

        if self.motion_packet:
            length_encoding = int(
                (sampled_length - self.length_range[0]) /
                (self.length_range[1] - self.length_range[0]) *
                (self.length_bins))
            direction_encoding = int(
                (sampled_slope_deg - self.dir_range[0]) /
                (self.dir_range[1] - self.dir_range[0]) *
                self.dir_bins)
            encoding = length_encoding + direction_encoding * self.length_bins
            encoding = np.asarray(encoding).reshape(1)

            encoding_packet = {'data': encoding}
            return [packet, encoding_packet]
        else:
            return [packet]

    def generateMotionBlurPSF(self, slope_deg, length):
        """Create a linear  motion blur PSF (Point Spread Function).

        Note
        -----
        The PSF image has always the odd size.

        Parameters
        ----------
        slope_deg : motion direction in degrees
        length : motion length in pixels

        Returns
        -------
        numpy.ndarray
            The computed PSF kernel
        """
        supersample_coef = 5
        supersample_thickness = supersample_coef

        if(length == 0.0):
            return np.ones((1, 1), dtype=float)

        int_length = np.ceil(length).astype(np.int)
        kernel_size_odd = (int_length / 2) * 2 + 1
        sup_kernel_size = np.rint(
            supersample_coef * kernel_size_odd).astype(np.int)

        sup_kernel = np.zeros(
            [sup_kernel_size, sup_kernel_size], dtype=np.float)

        sup_radius = (length * supersample_coef) / 2

        slope = slope_deg / 180 * np.pi
        pos = np.array([sup_radius, 0])
        R = np.array([
            [np.cos(-slope), -np.sin(-slope)],
            [np.sin(-slope), np.cos(-slope)]])
        pos = R.dot(pos)
        center = np.array(
            [float(sup_kernel_size) / 2,
             float(sup_kernel_size) / 2])

        cv2.line(
            sup_kernel,
            tuple(np.rint(center - pos).astype(np.int32)),
            tuple(np.rint(center + pos).astype(np.int32)),
            color=(1), thickness=supersample_thickness)

        psf = cv2.resize(
            sup_kernel, dsize=(int(kernel_size_odd), int(kernel_size_odd)),
            interpolation=cv2.INTER_AREA)

        #cv2.imshow('large', sup_kernel)
        psf = psf / psf.max()
        #cv2.imshow('small', psf)
        #cv2.waitKey()
        psf = psf / psf.sum()
        return psf


"""
def generatePSF(radius):
    scale = 25.0
    psfRadius = int(radius * scale + 0.5)
    center = int((int(radius)+2)*scale+scale/2)
    psf = np.zeros((2*center,2*center))
    cv2.circle(psf, (center,center), psfRadius, color=1.0, thickness=-1, lineType=cv2.CV_AA if 2 == int(cv2.__version__.split(".")[0]) else cv2.LINE_AA)
    psf = cv2.resize(psf, dsize=(0,0), fx=1.0/scale, fy=1.0/scale, interpolation=cv2.INTER_AREA)
    psf = psf / np.sum(psf)
    return psf, int(center/scale)


"""

def generateShakeBlur(RNG, startSamples=500, length=300, halflife=0.5, size=15):
    superSampling = 5

    # generate random acceleration
    a = RNG.randn(2, length + startSamples)
    # integrate speed
    a[0, :] = ewma(a[0, :], halflife=halflife * length)
    a[1, :] = ewma(a[1, :], halflife=halflife * length)

    # integrate position
    a = np.cumsum(a, axis=1)
    # skip first startSamples
    a = a[:, startSamples:]
    pathLength = np.sum(np.sum((a[:, 0:-1] - a[:, 1:])**2, axis=0)**0.5)
    # center the kernel
    a = a - np.mean(a, axis=1).reshape((2, 1))
    # cov = a.dot(a.T) / length
    # U, s, V = np.linalg.svd(cov)
    # a = a / s.max()**0.5 * size
    a = a / pathLength * size
    pathLength2 = np.sum(np.sum((a[:, 0:-1] - a[:, 1:])**2, axis=0)**0.5)
    print(pathLength, pathLength2)

    # normalize size
    maxDistance = np.max(np.abs(a))

    resolution = int(np.ceil(maxDistance) * 2 + 1)

    psf, t, t = np.histogram2d(
        a[0, :], a[1, :],
        bins=resolution * superSampling,
        range=[[-maxDistance, +maxDistance], [-maxDistance, +maxDistance]],
        normed=True)
    psf = cv2.resize(psf, (resolution, resolution), interpolation=cv2.INTER_AREA)
    psf = psf.astype(np.float32)
    psf = psf / np.sum(psf)
    return psf


"""
            # viewing transformations PSF, color, compression, noise, ...
            psfRadius = RNG.uniform(args.min_dof_radius, maxPSFRadius)
            psf, center = dataHelper.generatePSF(psfRadius)

            motionRadius = int(RNG.uniform(args.min_motion_radius, args.max_motion_radius) + 0.5)
            motionPSF = dataHelper.generateMotionBlur(RNG, resolution=motionRadius*2+1, halflife=args.motion_halflife) # can't skip to keep correct RNG state
            if motionRadius > 0:
                border = motionRadius + 1
                psf = cv2.copyMakeBorder(psf, border, border, border, border, cv2.BORDER_CONSTANT, value=0)
                psf = cv2.filter2D(psf, -1, motionPSF, borderType=cv2.BORDER_REPLICATE)

            if doMotionBlur:

            cropBlurNew = cv2.filter2D(cropOrigNew, cv2.CV_32F, psf, borderType=cv2.BORDER_REPLICATE)

            # remove border
            if args.border_size != 0:
                b = args.border_size
                cropOrigNew = cropOrigNew[b:-b,b:-b,:]
                cropBlurNew = cropBlurNew[b:-b,b:-b,:]

            noiseSdev = RNG.uniform(args.min_noise, maxNoise)
            cropBlurNew = dataHelper.addAdditiveWhiteNoise(RNG, cropBlurNew, noiseSdev)

"""
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
    '''
    Multiplies each color channel by random value.
    The formula for new collors is:
    c = c * 2**normal(std)

    Example:
    ColorBalance: {color_sdev: 0.02}
    '''
    def addParams(self):
        self.params.append(parameter(
            'color_sdev', required=False, default=0.02, parser=float,
            help='Standard deviation in color manipulation equations: c = c * 2**normal(std).'))

    def __init__(self, config):
        Configurable.__init__(self)
        self.log = logging.getLogger(__name__ + "." + type(self).__name__)
        self.addParams()
        self.parseParams(config)

    def __call__(self, packet, previous):
        img = np.copy(packet['data'])
        colorCoeff = [2**(np.random.standard_normal() * self.color_sdev) for x in range(img.shape[2])]
        for i, coef in enumerate(colorCoeff):
            img[:, :, i] *= coef
        packet['data'] = img
        return [packet]


class Noise(Configurable):
    '''
    Adds white additive noise.
    Noise standard deviation is uniformly sampled
    between min_noise and max_noise.

    Example:
    Noise: {max_noise:1}
    '''

    def addParams(self):
        self.params.append(parameter(
            'min_noise', required=False, default=0.0, parser=float,
            help='Minimum noise value.'))
        self.params.append(parameter(
            'max_noise', required=True, parser=float,
            help='Maximum noise value.'))

    def __init__(self, config):
        Configurable.__init__(self)
        self.log = logging.getLogger(__name__ + "." + type(self).__name__)
        self.addParams()
        self.parseParams(config)

    @staticmethod
    def addNoise(img, sdev):
        return img + np.random.randn(*img.shape) * sdev

    def __call__(self, packet, previous):
        noiseSdev = np.random.uniform(self.min_noise, self.max_noise)
        previous['op'] = functools.partial(Noise.addNoise, sdev=noiseSdev)
        packet['data'] = previous['op'](packet['data'])
        return [packet]


class ClipValues(Configurable):
    '''
    Clips values between min_val and max_val.

    Example:
    ClipValues: {min_val: 0,max_val: 1
    '''

    def addParams(self):
        self.params.append(parameter(
            'min_val', required=False, default=0.0, parser=float,
            help='Minimum clipping value.'))
        self.params.append(parameter(
            'max_val', required=False, default=1.0, parser=float,
            help='Maximum clipping value.'))

    def __init__(self, config):
        Configurable.__init__(self)
        self.log = logging.getLogger(__name__ + "." + type(self).__name__)
        self.addParams()
        self.parseParams(config)

    def __call__(self, packet, previous):
        packet['data'] = np.maximum(
            self.min_val, np.minimum(packet['data'], self.max_val))
        return [packet]


class GammaCorrection(Configurable):
    """
    Runs the image throug random gamma transformation:
    im = im**gamma / (255**gamma) * 255,
    where
    gamma = 2**normal(gamma_sdev)

    Example:
    GammaCorrection: {gamma_sdev:0.1}
    """
    def addParams(self):
        self.params.append(parameter(
            'gamma_sdev', required=True, parser=float,
            help='Gamma standard deviation.'))

    def __init__(self, config):
        Configurable.__init__(self)
        self.log = logging.getLogger(__name__ + "." + type(self).__name__)
        self.addParams()
        self.parseParams(config)

    def __call__(self, packet, previous):
        gamma = 2**(np.random.standard_normal() * self.gamma_sdev)
        packet['data'] = (packet['data'].astype(np.float32)**gamma) / (255**gamma) * 255
        return [packet]


class ReduceContrast(Configurable):
    """
    Can reduce contrast by randomly shifting zero intesity up
    (up to min_intensity) and shifting highest intensity down
    (at most to max_intensity).
    computes:
    img = img * (maxVal - minVal) + minVal

    Example:
    ReduceContrast: {min_intensity: -0.5, max_intensity: 0.5}
    """

    def addParams(self):
        self.params.append(parameter(
            'min_intensity', required=True, parser=float,
            help='Maximal minimum intensity.'))
        self.params.append(parameter(
            'max_intensity', required=True, parser=float,
            help='Maximal minimum intensity.'))

    def __init__(self, config):
        Configurable.__init__(self)
        self.log = logging.getLogger(__name__ + "." + type(self).__name__)
        self.addParams()
        self.parseParams(config)

    def __call__(self, packet, previous):
        minVal = np.random.uniform(0, self.min_intensity)
        maxVal = np.random.uniform(self.max_intensity, 255)
        packet['data'] = packet['data'] / 255.0 * (maxVal - minVal) + minVal
        return [packet]
