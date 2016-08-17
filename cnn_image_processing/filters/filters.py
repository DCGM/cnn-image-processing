'''
Filters
'''

from __future__ import division
from __future__ import print_function

import logging
import cv2
import numpy as np
import math
import copy

from .. utils import decode_dct
from .. utils import code_dct

# __all__ = ["TFilter", "TCropCoef8ImgFilter", "Crop", "LTCrop", "Label", "Mul",
#            "Div", "Add", "Sub", "JPGBlockReshape", "MullQuantTable", "Pass",
#            "Preview", "DecodeDCT", "CodeDCT", "Pad8", "PadCoefMirror"]


class TFilter(object):

    "Tuple Filter container."

    def __init__(self, filters=None):
        """
        TFileter constructor.
        Args:
            filters: list or tuple of filters.
        """
        self.filters = filters

    def n_filters(self):
        """
        Number of filters getter
        """
        if self.filters is not None:
            return len(self.filters)
        else:
            return 0

    def run(self, packets):
        """
        Runs all the filters with input packets.
        Args:
            packets: list or tuple of packets - same length as filters.
        """
        assert len(self.filters) == len(packets)
        flag = True
        l_result = [None] * len(self.filters)
        for i_filter, pfilter in enumerate(self.filters):
            l_result[i_filter] = pfilter(packets[i_filter])
            flag &= l_result[i_filter] is not None

        if flag:
            return tuple(l_result)
        else:
            return None

    def __call__(self, packets):
        return self.run(packets)


class THorizontalFilter(TFilter):

    '''
    Tuple horizontal passing filter etends the TFilter
    '''

    def __init__(self, filters=None):
        super(THorizontalFilter, self).__init__(filters)

    def run(self, packets):
        """
         Runs all the filters with input packets.
         Args:
             packets: list or tuple of packets - same length as filters.
         """
        assert len(self.filters) == len(packets)
        flag = True
        l_result = [None] * len(self.filters)
        hkwargs = {}

        for i_filter, pfilter in enumerate(self.filters):
            l_result[i_filter], hkwargs = pfilter(packets[i_filter], **hkwargs)
            flag &= l_result[i_filter] is not None

        if flag:
            return tuple(l_result)
        else:
            return None

    def __call__(self, packets):
        return self.run(packets)


class TCropCoef8ImgFilter(TFilter):

    """
    Specialized cropper for the n-tupple of jpeg-coef and n-image.
    Crop is obtained only from the image size divideable by 8.
    """

    def __init__(self, rng=None, crop_size=3, filters=None):
        """
        GenerateCropsFilter constructor
        Args:
          rng: Random state generator.
          crop_size: Is the crop size of the coef-jpeg data.
                For an image it is 8 x size.
        """
        super(TCropCoef8ImgFilter, self).__init__(filters)
        self.rng = rng if rng != None else np.random.RandomState(5)
        self.filters = filters
        self.crop_size = crop_size
        self.log = logging.getLogger(__name__ + ".TCropCoef8ImgFilter")

    def run(self, packets):
        """
        Generates the crop pivot and call all the crop filters in self.filters
        Args:
            data: The packets to be cropped from
        """
        assert len(packets) == len(self.filters)
        # always ommit the last row and column in the coef data
        shape = np.asarray(packets[0]['data'].shape[0:2]) - 1
        try:
            pivot = [self.rng.randint(0, dim - self.crop_size)
                     for dim in shape]
        except ValueError as val_err:
            path = packets[0]['path']
            self.log.error(" ".join((val_err.message, "Generate pivot", path)))
            return None
        pivot = np.asarray(pivot)
        crop_size = np.asarray((self.crop_size, self.crop_size))

        crops = []
        for filter_crop, packet in zip(self.filters, packets):
            crop_packet = filter_crop(packet, pivot, crop_size)
            crops.append(crop_packet)

        return tuple(crops)

    def __call__(self, data):
        return self.run(data)


class Crop(object):

    """
    Center pivot cropper.
    The center is is obtained as the floor divide op of size.
    Args:
        pivot: The coordinates of the pivot.
        size: The crop size
        scale: Scale factor of crop size, default 1.
        scale_pivot: Scale factor of pivot position.
    """

    def __init__(self, scale=1, scale_pivot=1):
        """
        Constructs the Crop - Center pivot Crop
        Args:
            scale: A scale factor the size is multiplied with.
                   Default = 1
            scale_pivot: A scale factor the pivot is multiplied with.
        """
        self.scale = scale
        self.scale_pivot = scale_pivot

    def crop(self, packet=None, pivot=None, size=None):
        """
        Crop the data with a center pivot and size. The pivot position
        and size may be scaled by the scale and scale_pivot factor.
        Args:
            data: Input to be cropped.
                numpy array
            pivot: (x,y) of the pivot.
                numpy array
            size: Size of the crop.
                int

        """
        p_y, p_x = pivot * self.scale_pivot
        size_y, size_x = (size * self.scale) // 2  # Floor divide op

        out_packet = {key: val for key, val in packet.items() if key != 'data'}

        out_packet['data'] = packet['data'][p_y - size_y:p_y + size_y,
                                            p_x - size_x:p_x + size_x]
        return out_packet

    def __call__(self, packet=None, pivot=None, size=None):
        return self.crop(packet=packet, pivot=pivot, size=size)


class LTCrop(object):

    """
    Left top pivot cropper.
    Args:
        pivot: The coordinates of the pivot.
        size: The crop size
        scale: Scale factor of the pivot position and crop size, default 1.
    """

    def __init__(self, scale=1):
        """
        Contructs the LTCrop - Left Top pivot Crop
        Args:
            scale: The scale factor the pivot and size is multiplied with.
                   Default = 1
        """
        self.scale = scale

    def crop(self, packet=None, pivot=None, size=None):
        """
        Crop the data with the top left pivot and size. The pivot position
        and size may be scaled by the scale factor.
        Args:
            data: Input to be cropped.
                numpy array
            pivot: (x,y) of the pivot.
                numpy array
            size: Size of the crop.
                int

        """
        p_y, p_x = pivot * self.scale
        size_y, size_x = size * self.scale

        out_packet = {key: val for key, val in packet.items() if key != 'data'}
        out_packet['data'] = packet['data'][p_y:p_y + size_y, p_x:p_x + size_x]
        out_packet['pivot'] = [p_y, p_x]

        return out_packet

    def __call__(self, packet=None, pivot=None, size=None):
        return self.crop(packet=packet, pivot=pivot, size=size)


class Label(object):

    """
    Label the data - creates the dictionary label_name: data
    """

    def __init__(self, name):
        """
        Create the simple Label filter which creates a dict name: data
        Args:
            name: Name of the label.
            string
        """
        self.label_name = name

    def label(self, packet):
        """
        Set the packet label key
        """
        packet['label'] = self.label_name
        return packet

    def __call__(self, packet):
        return self.label(packet)


class Mul(object):

    """
    Multiplies data with the given scalar.
    """

    def __init__(self, val=1):
        """
        Initialize the simple multiplier
        Args:
            val: The mul coefficient.
        """
        self.val = val

    def mul(self, packet):
        """
        Mul the packet data with defined val
        """
        packet['data'] *= self.val
        return packet

    def __call__(self, packet):
        return self.mul(packet)


class Div(object):

    """
    Divides data with the given scalar.
    """

    def __init__(self, val=1):
        """
        Initialize the simple multiplier
        Args:
            val: The div coefficient.
        """
        self.val = val

    def div(self, packet):
        """
        Div the packet data with val
        """
        packet['data'] /= self.val
        return packet

    def __call__(self, packet):
        return self.div(packet)


class Add(object):

    """
    Add an value from data.
    """

    def __init__(self, val=0):
        self.val = val

    def add(self, packet):
        """
        Add a val to packet data
        """
        packet['data'] += self.val
        return packet

    def __call__(self, packet):
        return self.add(packet)


class Sub(object):

    """
    Subtract an value from data.
    """

    def __init__(self, val=0):
        self.val = val

    def sub(self, packet):
        """
        Subtract a val from from packet data
        """
        packet['data'] -= self.val
        return packet

    def __call__(self, packet):
        return self.sub(packet)


class JPGBlockReshape(object):

    """
    Resample the input packet's data into the u/8 * x/8 * 64 dim data
    """

    def __init__(self):
        pass

    def reshape(self, packet):
        """
        Reshape the packet's data into [y/8, x/8, 64]
        """
        data = packet['data']
        assert data.shape[2] == 1
        shape = [dim // 8 for dim in data.shape[0:2]]
        shape.append(64)
        shaped_data = np.zeros(shape, dtype=np.float32)
        step = 8
        for y_id in xrange(shaped_data.shape[0]):
            for x_id in xrange(shaped_data.shape[1]):
                in_y = step * y_id
                in_x = step * x_id
                shaped_data[y_id, x_id] =\
                    data[in_y:in_y + step, in_x:in_x + step].reshape(64)

        packet['data'] = shaped_data
        return packet

    def __call__(self, packet):
        return self.reshape(packet)


class MulQuantTable(object):

    """
    Mul thepacket's data with its quant table stored in [y, x, 64:]
    """

    def __init__(self):
        pass

    def mull(self, packet):
        """
        Mul the packet's data stored in the [y, x, 0:64] with its quant table
        stored in [y, x, 64:]
        """
        data = packet['data']
        coef_quant_data = data[:, :, 0:64] * data[:, :, 64:]
        packet['data'] = coef_quant_data
        return packet

    def __call__(self, packet):
        return self.mull(packet)


class Pass(object):

    "Dummy object passing the data."

    def run(self, packet):
        """
        Return packet.
        """
        return packet

    def __call__(self, packet):
        return packet


class Preview(object):

    """
    Try to preview the packet's data as the image via OpenCV
    """

    def __init__(self, norm=1, shift=0, name=None):
        self.norm = norm
        self.shift = shift
        self.name = name

    def preview(self, packet):
        """
        Preview the packet's data as an image
        """
        name = None
        if self.name is None:
            if 'label' in packet:
                name = packet['label']
            else:
                name = packet['path']
        else:
            name = self.name

        img = packet['data'] / self.norm + self.shift
        cv2.imshow(name, img)
        cv2.waitKey(1000)
        return packet

    def __call__(self, packet):
        return self.preview(packet)


class DecodeDCT(object):

    """
    Decodes the coefs back into the pixels
    """

    def __init__(self):
        pass

    def decode(self, packet):
        """
        Decodes the image stored as coefs into the pixels and also reshape
        the result data back from [y/8, x/8, 64] to [y, x, 1]
        """
        packet['data'] = decode_dct(packet['data'])
        return packet

    def __call__(self, packet):
        return self.decode(packet)


class CodeDCT(object):

    """
    Code the image to coefs data
    """

    def __init__(self):
        pass

    def code(self, packet):
        """
        Ccode the image into coefs and reshape the data from [y, x, 1] to
        [y/8, x/8, 64]
        """
        packet['data'] = code_dct(packet['data'])
        return packet

    def __call__(self, packet):
        return self.code(packet)


class Pad8(object):

    """
    Pad the packet's most left nad bottom data to be divideable by 8
    """

    def pad(self, packet):
        """
        Pad the packet's data most left and bottom to be divideable by 8
        """
        data = packet['data']
        data_shape = data.shape[0:2]
        res = [dim % 8 for dim in data_shape]
        borders = [8 - rem if rem != 0 else 0 for rem in res]
        yx_borders = [(0, borders[0]), (0, borders[1]), (0, 0)]
        pad_data = np.pad(data, yx_borders, mode='edge')
        packet['orig_shape'] = data.shape
        packet['data'] = pad_data
        return packet

    def __call__(self, packet):
        return self.pad(packet)


class PadCoefMirror(object):

    '''
    Pad the packet's data representing coefficients by its mirrored view
    '''

    def __init__(self):
        '''
        Cnstructor - initialize the vertical, horizontal and corner swap masks
        '''
        self.pad_key = 'padding'

        self.horizontal = np.asarray([
            +1, -1, +1, -1, +1, -1, +1, -1,
            +1, -1, +1, -1, +1, -1, +1, -1,
            +1, -1, +1, -1, +1, -1, +1, -1,
            +1, -1, +1, -1, +1, -1, +1, -1,
            +1, -1, +1, -1, +1, -1, +1, -1,
            +1, -1, +1, -1, +1, -1, +1, -1,
            +1, -1, +1, -1, +1, -1, +1, -1,
            +1, -1, +1, -1, +1, -1, +1, -1
        ])
        self.vertical = np.asarray([
            +1, +1, +1, +1, +1, +1, +1, +1,
            -1, -1, -1, -1, -1, -1, -1, -1,
            +1, +1, +1, +1, +1, +1, +1, +1,
            -1, -1, -1, -1, -1, -1, -1, -1,
            +1, +1, +1, +1, +1, +1, +1, +1,
            -1, -1, -1, -1, -1, -1, -1, -1,
            +1, +1, +1, +1, +1, +1, +1, +1,
            -1, -1, -1, -1, -1, -1, -1, -1
        ])
        self.corner = np.asarray([
            +1, -1, +1, -1, +1, -1, +1, -1,
            -1, +1, -1, +1, -1, +1, -1, +1,
            +1, -1, +1, -1, +1, -1, +1, -1,
            -1, +1, -1, +1, -1, +1, -1, +1,
            +1, -1, +1, -1, +1, -1, +1, -1,
            -1, +1, -1, +1, -1, +1, -1, +1,
            +1, -1, +1, -1, +1, -1, +1, -1,
            -1, +1, -1, +1, -1, +1, -1, +1
        ])

    def pad(self, packet):
        '''
        Mirror the edge coefficients 64 channel vector representing patch of
        8x8 pixels
        '''
        padding = [(1, 1), (1, 1), (0, 0)]
        padding_ar = np.asarray(padding, np.int)

        if self.pad_key in packet:
            packet[self.pad_key] += padding_ar
        else:
            packet[self.pad_key] = padding_ar

        pad_data = np.pad(
            packet['data'], padding, mode='edge')

        for patch in pad_data[1:-1, 0]:
            patch = patch * self.horizontal
        for patch in pad_data[1:-1, -1]:
            patch *= self.horizontal

        for patch in pad_data[0, 1:-1]:
            patch *= self.vertical
        for patch in pad_data[-1, 1:-1]:
            patch *= self.vertical

        pad_data[0, 0] *= self.corner
        pad_data[0, -1] *= self.corner
        pad_data[-1, 0] *= self.corner
        pad_data[-1, -1] *= self.corner

        packet['data'] = pad_data
        return packet

    def __call__(self, packet):
        return self.pad(packet)


class JPEG(object):

    '''
    JPEG filter compress the input data with specified quality
    '''

    def __init__(self, quality=20):
        '''
        JPEG constructor

        args:
            quality: int
                The JPEG compression quality
        '''
        self.qual = quality

    def compress(self, packet):
        '''
        Compress the input packet's data with the jpeg encoder
        '''
        res, img_str = cv2.imencode('.jpeg', packet['data'],
                                    [cv2.IMWRITE_JPEG_QUALITY, self.qual])
        assert res is True
        img = cv2.imdecode(np.asarray(bytearray(img_str), dtype=np.uint8),
                           cv2.IMREAD_UNCHANGED)
        if len(img.shape) == 2:
            img = img.reshape(img.shape[0], img.shape[1], 1)

        packet['data'] = img
        return packet

    def __call__(self, packet):
        return self.compress(packet)


class ShiftImg(object):

    '''
    Shift - move packet's data - image
    '''

    def __init__(self, move_vec=None, mode=None, rng_seed=5):
        '''
        Shift the image data by the move_vec
        First, the image is padded by the symmetric copy of its edge,
        second the crop is taken of such padded image to obtain the same
        sized image but moved by the move_vec
        The move could be fixed or randomly sampled from the uniform dist
        defined by the move_vec (0, move_vec[0]) for the axe 0 etc.

        args:
            move_vec: list
                amount of the data shift, sign defines the direction, where
                + means shift to left, - shift to right
            mode: string
                mode of the shift:
                    None: default shift according the move_vec
                    'random': randomly sample from (0, move_vec) with uniform
                              distribution the amount of shift according the
                              move_vec
            rng: numpy random state generator
                the random state generator seed, default is 5
        '''
        self.mode = mode
        if move_vec is None:
            self.move_vec = np.zeros(3)
        else:
            self.move_vec = np.asarray(move_vec)
        self.rng = np.random.RandomState(rng_seed)
        assert not (self.mode == 'random' and
                    all(val == 0 for val in self.move_vec))

    def shift(self, packet, **kwargs):
        '''
        Shift data according the move vector with a mirror padding
        '''
        padding = []
        move_vec = None
        mode = None
        abs_move_vec = None
        move_vec_out = None

        # Use the horizontal args - kwargs if any
        if 'move_vec' in kwargs:
            move_vec = kwargs['move_vec']
            abs_move_vec = np.abs(move_vec)
            mode = None
        else:
            move_vec = self.move_vec
            abs_move_vec = np.abs(self.move_vec)
            mode = self.mode
            move_vec_out = np.copy(self.move_vec)
            kwargs['move_vec'] = move_vec_out

        for i_val, val in enumerate(abs_move_vec):
            if val == 0:
                padding.append((0, 0))
                continue

            if mode == 'random':
                val = self.rng.randint(0, val)
                move_vec_out[i_val] = val * np.sign(move_vec[i_val])

            if move_vec[i_val] >= 0:
                padding.append((0, val))
            else:
                padding.append((val, 0))

        pad_data = np.pad(packet['data'], padding, mode='symmetric')
        start_y = padding[0][1]
        start_x = padding[1][1]
        shape_y, shape_x, _ = packet['data'].shape
        packet['data'] = pad_data[start_y:start_y + shape_y,
                                  start_x:start_x + shape_x]

        return packet, kwargs

    def __call__(self, packet, **kwargs):
        return self.shift(packet, **kwargs)

class Resize(object):

    """
    Resize image.
    """

    def __init__(self, scale=0.5):
        self.scaleFactor = scale

    def scale(self, packet):
        """
        Resize image by scale - using cv2.INTER_AREA
        """
        packet['data'] = cv2.resize(packet['data'], (0,0), fx=self.scaleFactor, fy=self.scaleFactor, interpolation=cv2.INTER_NEAREST)
        shape = packet['data'].shape
        packet['data'] = np.reshape( packet['data'], (shape[0], shape[1], -1))
        return packet

    def __call__(self, packet):
        return self.scale(packet)


class Round(object):

    """
    Round values.
    """

    def __init__(self):
        self.i = 1

    def round(self, packet):
        """
        Subtract a val from from packet data
        """
        packet['data'] = np.round(packet['data'])
        return packet

    def __call__(self, packet):
        return self.round(packet)

class FixedCrop(object):

    """
    Crop center portion of the image.
    """

    def __init__(self, size=10):
        self.size = size

    def crop(self, packet):
        """
        Crop center portion of the image.
        """
        border = (int((packet['data'].shape[0]-self.size)/2), int((packet['data'].shape[1]-self.size)/2))
        packet['data'] = packet['data'][ border[0]:border[0]+self.size, border[1]:border[1]+self.size]
        return packet

    def __call__(self, packet):
        return self.crop(packet)


class TRandomCropper(object):

    """
    Crop random area around an annotated position
    """

    def __init__(self, size, random_seed=1):
        self.RNG = np.random.RandomState(random_seed)
        self.size = size()


def warpPerspective(rx, ry, rz, fov, img, positions=None, shift=(0,0)):

    s = max(img.shape[0:2])
    rotVec = np.asarray((rx*np.pi/180,ry*np.pi/180, rz*np.pi/180))
    rotMat, j = cv2.Rodrigues(rotVec)
    rotMat[0, 2] = 0
    rotMat[1, 2] = 0
    rotMat[2, 2] = 1

    f = 0.3
    trnMat1 = np.asarray(
        (1, 0, -img.shape[1]/2,
         0, 1, -img.shape[0]/2,
         0, 0, 1)).reshape(3, 3)

    T1 = np.dot(rotMat, trnMat1)
    distance = (s/2)/math.tan(fov*np.pi/180)
    T1[2, 2] += distance

    cameraT = np.asarray(
        (distance, 0, img.shape[1]/2 + shift[1],
         0, distance, img.shape[0]/2 + shift[0],
         0, 0, 1)).reshape(3,3)

    T2 = np.dot(cameraT, T1)

    newImage = cv2.warpPerspective(img, T2, (img.shape[1], img.shape[0]), borderMode=cv2.BORDER_REPLICATE, flags=cv2.INTER_CUBIC)
    if positions is None:
        return newImage
    else:
        return newImage, np.squeeze( cv2.perspectiveTransform(positions[None, :, :], T2), axis=0)

class VirtualCamera(object):
    """
    Transform image with random camera. X and Y are in the image plane, Z point to the camera. maxShift is in pixels. The rotation deviations are in degrees.
    """

    def __init__(self, rotationX_sdev, rotationY_sdev, rotationZ_sdev, min_fov, max_fov, max_shift):
        self.rotationX_sdev = rotationX_sdev
        self.rotationY_sdev = rotationY_sdev
        self.rotationZ_sdev = rotationZ_sdev
        self.min_fov = min_fov
        self.max_fov = max_fov
        self.max_shift = max_shift

    def __call__(self, packet):
        #geometric transformations
        rx = np.random.standard_normal() * self.rotationX_sdev
        ry = np.random.standard_normal() * self.rotationY_sdev
        rz = np.random.standard_normal() * self.rotationZ_sdev
        fov = np.random.uniform(self.min_fov, self.max_fov)
        shift = np.random.standard_normal(2)
        shift = shift / np.sum(shift**2)**0.5 * np.random.uniform(self.max_shift)

        packet['data'] = warpPerspective(rx, ry, rz, fov, packet['data'], shift=shift)

        return packet

class Noise(object):
    def __init__(self, min_noise, max_noise):
        self.min_noise = min_noise
        self.max_noise = max_noise

    def __call__(self, packet):
        noiseSdev = np.random.uniform(self.min_noise, self.max_noise)
        packet['data'] = packet['data'] + np.random.randn(*packet['data'].shape) * noiseSdev
        return packet


class ColorBalance(object):
    def __init__(self, color_sdev):
        self.color_sdev = color_sdev
    def __call__(self, packet):
        img = packet['data']
        colorCoeff = [2**(np.random.standard_normal()*self.color_sdev) for x in range(img.shape[2])]
        for i, coef in enumerate(colorCoeff):
            img[:, :, i] *= coef
        packet['data'] = img
        return packet


class ReduceContrast(object):
    """
    Can reduce contrast by randomly shifting zero intesity up (up to min_intensity) and
    shifting highest intensity down (at most to max_intensity).
    """
    def __init__(self, min_intensity, max_intensity):
        self.min_intensity = min_intensity
        self.max_intensity = max_intensity

    def __call__(self, packet):
        minVal = np.random.uniform(0, self.min_intensity)
        maxVal = np.random.uniform(self.max_intensity, 255)
        packet['data'] = packet['data']/255.0 * (maxVal-minVal) + minVal
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
        gamma = 2**(np.random.standard_normal()*self.gamma_sdev)
        packet['data'] = (packet['data'].copy().astype(np.float32)**gamma) / (255**gamma) * 255
        return packet

class Flip(object):
    """
    Flips image randomly in horizontal (or vertical) direction.
    """
    def __init__(self, horizontal, vertical):
        self.horizontal = horizontal
        self.vertical = vertical
    def __call__(self, packet):
        if self.horizontal and np.random.randint(2):
            packet['data'] = np.fliplr(packet['data'])
        if self.vertical and np.random.randint(2):
            packet['data'] = np.flipud(packet['data'])
        return packet


class Copy(object):
    """
    Just create a data copy
    """
    def __init__(self):
        pass
    def __call__(self, packet):
        old = packet['data']
        packet = copy.copy(packet)
        packet['data'] = np.copy(packet['data'])
        newPacket = {}
        return packet

class ClipValues(object):
    def __init__(self, minVal=0, maxVal=255):
        self.minVal = minVal
        self.maxVal = maxVal
        pass
    def __call__(self, packet):
        packet['data'] = np.maximum( np.minimum(packet['data'], self.maxVal), self.minVal)
        return packet
