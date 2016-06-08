from __future__ import print_function
import cv2
import logging
import numpy as np

class TFilter(object):
    "Tuple Filter container."
    def __init__(self, filters=None):
        """
        TFileter constructor.
        Args:
            filters: list or tuple of filters.
        """
        self.filters=filters
    
    def size(self):
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

class Pass(object):
    "Dummy object passing the data."
    
    def run(self, packet):
        """
        Return packet.
        """
        return packet
    
    def __call__(self, packet):
        return packet

class TCropCoef8ImgFilter(object):
    """
    Specialized cropper for the n-tupple of jpeg-coef and n-image.
    Crop is obtained only from the image size divideable by 8.
    """
    def __init__(self, rng=None, size=3, filters=None):
        """
        GenerateCropsFilter constructor
        Args:
          rng: Random state generator.
          size: Is the crop size of the coef-jpeg data.
                For an image it is 8 x size.
        """
        self.rng = rng if rng != None else np.random.RandomState(5)
        self.filters = filters
        self.size = size
  
    def run(self, packets):
        """
        Generates the crop pivot and call all the crop filters in self.filters
        Args:
            data: The packets to be cropped from
        """
        assert len(packets) == len(self.filters)
        # always ommit the last row and column in the coef data
        shape = np.asarray(packets[0]['data'].shape[0:2])-1
        pivot = [ self.rng.randint(0, dim - self.size) for dim in shape ]
        pivot = np.asarray(pivot)
        size = np.asarray((self.size, self.size))
        
        crops = []
        for filter_crop, packet in zip(self.filters, packets):
            crop_packet = filter_crop(packet, pivot, size)
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
        Contructs the Crop - Cemter pivot Crop
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
            scale: The scale factor the pivot ans size is multiplied with.
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
