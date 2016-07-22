'''
Filters
'''

from __future__ import division
from __future__ import print_function

import logging
from collections import namedtuple
from itertools import cycle
import cv2
import numpy as np

from .. utils import decode_dct
from .. utils import code_dct

__all__ = [
    "Packet", "FilterTArg", "FileListReader", "TFilter", "THorizontalFilter", "TCropCoef8ImgFilter", "Crop", "LTCrop", "Label", "Mul",
    "Div", "Add", "Sub", "JPGBlockReshape", "MulQuantTable", "Pass",
    "Preview", "DecodeDCT", "CodeDCT", "Pad8", "PadCoefMirror", "JPEG", "ShiftImg"]


class Packet(object):

    '''
    Packet is the class (more like struct from c++) to hold several attributes
    For the design pattern behind this approach, see
    http://python-3-patterns-idioms-test.readthedocs.io/en/latest/Messenger.html

    Also have a look at http://stackoverflow.com/a/610923 about: "easier to
    ask for forgiveness than permission" (EAFP) rather than "look before you
    leap" (LBYL) when dealing with the Packe attributres.
    '''

    def __init__(self, **kwargs):
        '''
        Packet intialize - set the attributes
        '''
        self.__dict__ = kwargs

FilterTArg = namedtuple('FilterTArg', ['packet', 'arg'])


class FileListReader(object):

    '''
    Reads the file once or in loop. Line is plit according blank chars into
    segments. All segments are sent in a vector of corresponing packets
    of type Packet.
    '''

    def __init__(self, data=None, loop=False):
        '''
        Inititlize the FileListReader

        Read the file into the list (default) in case of loop is False,
        otherwise the loop is True the file is buffered into the cycle buffer
        to repeatedly read data from.

        args:
            data: str
                list with the data (usualy paths to data)
            loop: Boolean
                loop reads once in case loop is False (default), loop otherwise
        '''
        self.data = data
        self.loop = loop
        self.buf = None
        self.iter = None
        self.log = logging.getLogger(".".join([__name__, type(self).__name__]))

    def init_data(self, data=None):
        '''
        Initilize the data reader
        '''
        if data is not None:
            self.data = data

        try:
            with open(self.data, 'r') as flist:
                self.buf = []
                for line in flist:
                    self.buf.append(line)

            if self.loop is True:
                self.iter = cycle(self.buf)
            else:
                self.iter = iter(self.buf)

        except EnvironmentError as enver:
            self.log.exception("Failed to read: %s", self.data)
            raise enver

    def read(self):
        '''
        Read and parse next line in the buffered file and return packets

        Packets is a vector of packets (per segment in one line)
        '''

        ltargs = None
        try:
            line = next(self.iter)
            ltargs = [FilterTArg(Packet(path=segment), None)
                      for segment in line.split()]

            return ltargs

        except StopIteration:
            self.log.debug("Read end. %s", self.data)
            raise

    def __call__(self, args):
        '''
        Note: args are here only to be aligned with call conv of other
        filters.
        '''
        return self.read()


class Preview(object):

    """
    Sow the data as image via OpenCV
    """

    def __init__(self, norm=1, shift=0, name=None):
        '''
        Initilize the Preview

        args:
         norm: float
            Value to divide the data per element with
        shift: float
            Value to shif the data with
        name: str
            Name of the preview window
            In case it is not defined the looks for packet.label or packet.path
        '''
        self.norm = norm
        self.shift = shift
        self.name = name

    def preview(self, targs):
        """
        Preview the packet's data as an image
        """

        packet = targs.packet
        name = None
        if self.name is not None:
            name = self.name
        elif hasattr(packet, 'label'):
            name = packet.label
        else:
            name = packet.path

        img = packet.data / self.norm + self.shift
        cv2.imshow(name, img)
        cv2.waitKey(10)

        return targs

    def __call__(self, packet):
        return self.preview(packet)


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

    def run(self, targs):
        """
        Return packet.
        """
        return targs

    def __call__(self, targs):
        return targs


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
