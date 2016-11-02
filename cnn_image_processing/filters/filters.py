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
import time

from .. utils import decode_dct
from .. utils import code_dct

# __all__ = ["TFilter", "TCropCoef8ImgFilter", "Crop", "LTCrop", "Label", "Mul",
#            "Div", "Add", "Sub", "JPGBlockReshape", "MullQuantTable", "Pass",
#            "Preview", "DecodeDCT", "CodeDCT", "Pad8", "PadCoefMirror"]

from ..utilities import parameter, Configurable, ContinuePipeline, TerminatePipeline


class Buffer(Configurable):
    """
    FIFO buffer with random selection.
    Use in conjuction with HorizontalPassPackets to buffer
    multiple pipeline streams.

    Example:
        HorizontalPassPackets: {pass_through: False}
        HorizontalPassPackets: {pass_through: False}
        Buffer: {size: 500, rng_seed: 24}
    """

    def addParams(self):
        self.params.append(parameter(
            'size', required=True,
            parser=lambda x: max(int(x), 1),
            help='Size of the buffer'))
        self.params.append(parameter(
            'rng_seed', required=False, default=None,
            parser=lambda x: max(int(x), 1),
            help='Size of the buffer'))

    def __init__(self, config):
        Configurable.__init__(self)
        self.log = logging.getLogger(__name__ + "." + type(self).__name__)
        self.addParams()
        self.parseParams(config)
        self.buffer = [None] * self.size
        self.filled = 0
        self.bufferHead = 0
        if self.rng_seed:
            self.rng = np.random.RandomState(self.rng_seed)
        else:
            self.rng = np.random.RandomState()

    def add(self, packets):
        self.buffer[self.bufferHead] = packets
        self.filled = min(self.filled + 1, self.size)
        self.bufferHead = (self.bufferHead + 1) % self.size

    def getRandom(self):
        if not self.filled:
            self.log.warning('Empty packet buffer.')
            raise ContinuePipeline
        pos = self.rng.randint(self.filled)
        return copy.copy(self.buffer[pos])

    def __call__(self, packet, previous):
        if packet:
            packets = []
            if 'packets' in previous:
                packets.extend(previous['packets'])
            packets.append(packet)
            self.add(packets)

        return self.getRandom()




class Crop(Configurable):
    """
    Fixed region cropper.
    The pivot is is obtained as the floor divide of size.
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


class Label(Configurable):
    """
    Label the packets - adds label_name: data

    Example:
    Label: {label_name: "Some label"}
    """
    def addParams(self,):
        self.params.append(parameter(
            'label_name', required=True, help='The label value.', parser=str))

    def __init__(self, config):
        Configurable.__init__(self)
        self.log = logging.getLogger(__name__ + "." + type(self).__name__)
        self.addParams()
        self.parseParams(config)

    def __call__(self, packet, previous):
        packet['label'] = self.label_name
        return [packet]

class MulAdd(Configurable):
    """
    Multiplies data with the given scalar and adds another scalar.
    Multiplies first, adds second.
    Supports horizontal operation propagation.

    Example:
    MulAdd: {'mul': 0.004, 'add': -0.5}
    """
    def addParams(self,):
        self.params.append(parameter(
            'mul', required=False, default=1.0, parser=float, help='Multiplication factor.'))
        self.params.append(parameter(
            'add', required=False, default=0.0, parser=float, help='Additive factor.'))

    def __init__(self, config):
        Configurable.__init__(self)
        self.log = logging.getLogger(__name__ + "." + type(self).__name__)
        self.addParams()
        self.parseParams(config)
        self.operation = lambda x: x * self.mul + self.add

    def __call__(self, packet, previous):
        packet['data'] = self.operation(packet['data'])
        previous['op'] = self.operation
        return [packet]


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


class Pass(Configurable):
    """
    Dummy filter passing the data forward.

    Example:
    Pass: {}
    """

    def __init__(self, config):
        Configurable.__init__(self)
        self.log = logging.getLogger(__name__ + "." + type(self).__name__)
        self.addParams()
        self.parseParams(config)

    def __call__(self, packet, previous):
        return [packet]


class Preview(Configurable):
    """
    Try to preview the packet's data as the image via OpenCV

    Example:
    Preview: {norm=1, shift=0, name='Data'}
    """

    def addParams(self):
        self.params.append(parameter(
            'norm', required=False, default=1.0, parser=float,
            help='Image will be multiplied by this number.'))
        self.params.append(parameter(
            'shift', required=False, default=0, parser=float,
            help='This number will be added to the image after the multiplication.'))
        self.params.append(parameter(
            'name', required=False, default=None, parser=str,
            help='Name of a window used to display the images. Uses packet["label"] in not specified'))

    def __init__(self, config):
        Configurable.__init__(self)
        self.log = logging.getLogger(__name__ + "." + type(self).__name__)
        self.addParams()
        self.parseParams(config)

    def __call__(self, packet, previous):
        """
        Preview the packet's data as an image
        """
        name = self.name
        if name is None and 'label' in packet:
            name = packet['label']
        img = packet['data'] / self.norm + self.shift
        cv2.imshow(name, img)
        cv2.waitKey(1)

        return [packet]


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

    def __init__(self, config):
        Configurable.__init__(self)
        self.log = logging.getLogger(__name__ + "." + type(self).__name__)
        self.addParams()
        self.parseParams(config)

    def __call__(self, packet, previous):
        '''
        Compress the input packet's data with the jpeg encoder
        '''
        res, img_str = cv2.imencode('.jpeg', packet['data'],
                                    [cv2.IMWRITE_JPEG_QUALITY, self.quality])
        if not res:
            self.log.error('Failed to JPEG encode data.')
            raise ContinuePipeline

        img = cv2.imdecode(np.asarray(bytearray(img_str), dtype=np.uint8),
                           cv2.IMREAD_UNCHANGED)
        packet['data'] = img.reshape(img.shape[0], img.shape[1], -1)
        return [packet]


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


class Resize(Configurable):
    """
    Resize image using cv2.INTER_AREA.

    Examples:
    Resize: {fixedSize=[100,100]}
    Resize: {scale=0.5}
    """
    def addParams(self):
        self.params.append(parameter(
            'fixedSize', required=False, default=None,
            parser=lambda x: tuple([max(1, int(i)) for i in x[0:2]]),
            help='Pair of integers [with, height].'))
        self.params.append(parameter(
            'scale', required=False, default=None,
            parser=lambda x: max(0.0, float(x)),
            help='Image scale.'))

    def __init__(self, config):
        Configurable.__init__(self)
        self.log = logging.getLogger(__name__ + "." + type(self).__name__)
        self.addParams()
        self.parseParams(config)

        if self.fixedSize is None and self.scale is None:
            self.log.error('Either "scale" or "fixedSize" has to be specified.')
            raise ValueError
        if self.fixedSize is not None and self.scale is not None:
            self.log.warning('Both "scale" and "fixedSize" can not be specified at the same time.')
            raise ValueError

    def __call__(self, packet, previous):
        if self.scale:
            packet['data'] = cv2.resize(packet['data'], (0, 0), fx=self.scale, fy=self.scale, interpolation=cv2.INTER_NEAREST)
        else:
            packet['data'] = cv2.resize(packet['data'], self.fixedSize, interpolation=cv2.INTER_AREA)
        shape = packet['data'].shape
        packet['data'] = np.reshape(packet['data'], (shape[0], shape[1], -1))
        return [packet]


class CustomFunction(Configurable):
    """
    Allows you to write a custom lambda function which is applied
    to packet['data'] (which is most probably numpy array wiht dimensions
    [height, width, channels]).
    You can use numpy as np and OpenCV as cv2.

    Example:
    CustomFunction: {function: 'lambda x: (x / 10 + 15).astype(np.float32)' }
    """
    def addParams(self):
        self.params.append(parameter(
            'function', required=True, parser=str,
            help='Lambda function that will be applied.'))

    def __init__(self, config):
        Configurable.__init__(self)
        self.log = logging.getLogger(__name__ + "." + type(self).__name__)
        self.addParams()
        self.parseParams(config)
        self.op = eval(self.function)
        if not callable(self.op):
            self.log.error('The specified function has to be lambda function or something else callable.')
            raise ValueError
        import inspect
        args = inspect.getargspec(self.op)
        if len(args.args) != 1:
            self.log.error('The specified function has to take only one argument.')
            raise ValueError

    def __call__(self, packet, previous):
        packet['data'] = self.op(packet['data'])
        previous['op'] = self.op
        return [packet]


class Round(Configurable):
    """
    Round values.

    Example:
    Round: {}
    """
    def __init__(self, config):
        Configurable.__init__(self)
        self.log = logging.getLogger(__name__ + "." + type(self).__name__)
        self.addParams()
        self.parseParams(config)

    def __call__(self, packet, previous):
        packet['data'] = np.round(packet['data'])
        return [packet]

class CentralCrop(Configurable):
    """
    Crop center portion of the image.

    Example:
    CentralCrop: {size=10}
    """
    def addParams(self):
        self.params.append(parameter(
            'size', required=True,
            parser=lambda x: max(int(x), 1),
            help='Size of the croped region.'))

    def __init__(self, config):
        Configurable.__init__(self)
        self.log = logging.getLogger(__name__ + "." + type(self).__name__)
        self.addParams()
        self.parseParams(config)

    def __call__(self, packet, previous):
        border = (int((packet['data'].shape[0] - self.size) / 2),
                  int((packet['data'].shape[1] - self.size) / 2))
        packet['data'] = packet['data'][
            border[0]:border[0] + self.size,
            border[1]:border[1] + self.size]
        return packet


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


class HorizontalPassPackets(Configurable):
    """
    Does nothing - only passes packets to filters in the same pipeline stage.
    pass_through - if true, passes packets down the pipeline

    Example:
    HorizontalPassPackets: {pass_through=True}
    """

    def addParams(self):
        self.params.append(parameter(
            'pass_through', required=False, default=True,
            parser=bool,
            help='If true, passes packets down the pipeline.'))

    def __init__(self, config):
        Configurable.__init__(self)
        self.log = logging.getLogger(__name__ + "." + type(self).__name__)
        self.addParams()
        self.parseParams(config)

    def __call__(self, packet, previous):
        if 'packets' not in previous.keys():
            previous['packets'] = []
        previous['packets'].append(packet)
        if self.pass_through:
            return [packet]
        else:
            return []
