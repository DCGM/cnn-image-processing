from __future__ import print_function
import cv2
import logging
import numpy as np

from ..utils import decode_dct

class TFilter(object):
    "Tuple Filter container."
    def __init__(self, filters=None):
        """
        TFileter constructor.
        Args:
            filters: list or tuple of filters.
        """
        self.filters=filters
    
    def n_filters(self):
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
        l_result = [None]*len(self.filters)
        for i_filter, pfilter in enumerate(self.filters):
            l_result[i_filter] = pfilter(packets[i_filter])
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
        shape = np.asarray(packets[0]['data'].shape[0:2])-1
        try:
            pivot = [ self.rng.randint(0, dim - self.crop_size) for dim in shape ]
        except ValueError as ve:
            path = packets[0]['path']
            self.log.error(" ".join( (ve.message, "Generate pivot", path) ) )
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
        size_y, size_x = (size * self.scale) // 2 # Floor divide op
        
        out_packet = {key:val for key, val in packet.items() if key != 'data'}
        
        out_packet['data'] = packet['data'][p_y-size_y:p_y+size_y,
                                            p_x-size_x:p_x+size_x]
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
        
        out_packet = {key:val for key, val in packet.items() if key != 'data'}
        out_packet['data'] = packet['data'][p_y:p_y+size_y, p_x:p_x+size_x]
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
        packet['data'] *= self.val
        return packet
    
    def __call__(self, packet):
        return self.mul(packet)

class Add(object):
    """
    Add an value from data.
    """
    def __init__(self, val=0):
        self.val = val
        
    def add(self, packet):
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
        packet['data'] -= self.val
        return packet
    
    def __call__(self, packet):
        return self.sub(packet)

class JPGBlockReshape(object):
    def __init__(self):
        pass
    
    def reshape(self, packet):
        data = packet['data']
        assert(data.shape[2] == 1)        
        shape = [ dim // 8 for dim in data.shape[0:2] ]
        shape.append(64)
        shaped_data = np.zeros( shape, dtype=np.float32)
        step = 8
        for y_id in xrange(shaped_data.shape[0]):
            for x_id in xrange(shaped_data.shape[1]):
                in_y = step*y_id
                in_x = step*x_id 
                shaped_data[y_id, x_id] =\
                    data[in_y:in_y+step, in_x:in_x+step].reshape(64)
    
        packet['data'] = shaped_data
        return packet
    
    def __call__(self, packet):
        return self.reshape(packet)
    
class MullQuantTable(object):
    def __init__(self):
        pass
    
    def mull(self, packet):
        data = packet['data']
        coef_quant_data = data[:,:,0:64] * data[:,:,64:]
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
    def __init__(self, scale=1, shift=0, name=None):
        self.scale = scale
        self.shift = shift
        self.name = name
        
    def preview(self, packet):
        name = None
        if self.name is None:
            if 'label' in packet:
                name = packet['label']
            else:
                name = packet['path']
        else: name = self.name
        
        img = packet['data'] * self.scale + self.shift
        cv2.imshow(name, img)
        cv2.waitKey()
        return packet
        
    def __call__(self, packet):
        return self.preview(packet)

class DecodeDCT(object):
    def __init__(self):
        pass
    def decode(self, packet):
        packet['data'] = decode_dct(packet['data'])
        return packet
    
    def __call__(self, packet):
        return self.decode(packet)
