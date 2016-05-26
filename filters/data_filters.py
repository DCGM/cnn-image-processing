#!/usr/bin/env python
from __future__ import print_function
import cv2
import logging
import numpy as np

module_logger = logging.getLogger(__name__)

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
        
    def run(self, data):
        """
        Runs all the filters with input data.
        Args:
            data: list or tuple of data - same length as filters.
        """
        assert len(self.filters) == len(data)
        flag = True
        l_result = [None]*len(self.filters)
        for i_ftr, ftr in enumerate(self.filters):
            l_result[i_ftr] = ftr(data[i_ftr])
            flag &= l_result[i_ftr] is not None
        
        if flag:   
            return tuple(l_result)
        else:
            return None
    
    def __call__(self, data):
        return self.run(data)

class Pass(object):
    "Dummy object passing the data."
    
    def run(self, data):
        """
        Return data.
        """
        return data
    
    def __call__(self, data):
        return data

class MotionBlurPSF:
    """
    ToDo modify the code to fit into the Filter like usage
    """
    def __init__(self, slope_deg, length):
        self.slope_deg = slope_deg
        self.length = length
    
    def generate(self, RNG, slope_deg = None, length = None):
        """
        Compute the motion blur PSF (Point Spread Function).
        
        Note
        -----
        The PSF has always the odd size.
        
        Parameters
        ----------
        RNG : numpy.random.RandomState() object
            Random state for sampling the slope_deg
        slope_deg : numpy.array
            the half open [low, high) interval of slope of the motion vector
            related to x axe in degrees to uniformly sample from
        length : numpy.array
            the half open [low, high) interval of motion vector length in pixels
            to uniformly sample from
        
        Returns
        -------
        numpy.ndarray
            The computed PSF kernel
        float
            sampled slope deg
        float
            sampled length
        """
        slope_deg = self.slope_deg if slope_deg is None else slope_deg
        length = self.length if length is None else length
        supersample_coef = 100
        supersample_thickness = 100 / 10
        sampled_slope_deg = RNG.uniform(low=slope_deg[0], high=slope_deg [1])
        sampled_length = RNG.uniform(low=length[0], high=length [1])
        
        if(sampled_length == 0.0):
            return np.ones((1, 1), dtype=float)
        
        int_sampled_length = np.ceil(sampled_length).astype(np.int)
        kernel_size_odd = int_sampled_length + 1 if(int_sampled_length % 2 == 0) else int_sampled_length
        int_sup_sampled_length = np.rint(supersample_coef * sampled_length).astype(np.int)
        kernel_sup_size_odd = int(int_sup_sampled_length + 1 if (int_sup_sampled_length % 2 == 0) else int_sup_sampled_length)
        
        kernel_supersample = np.zeros([kernel_sup_size_odd, kernel_sup_size_odd], dtype=np.float)
        v_center_sup_kernel = (int(kernel_sup_size_odd / 2.), int(kernel_sup_size_odd / 2.))
        cv2.line(kernel_supersample, (0, v_center_sup_kernel[1]), (kernel_sup_size_odd - 1, v_center_sup_kernel[1]), color=(1), thickness=int(supersample_thickness * sampled_length))
        rot_mat = cv2.getRotationMatrix2D(center=v_center_sup_kernel, angle=sampled_slope_deg, scale=1)
        
        psf = cv2.warpAffine(src=kernel_supersample, M=rot_mat, dsize=kernel_supersample.shape)
        psf = cv2.resize(psf, dsize=(kernel_size_odd, kernel_size_odd), fx=0, fy=0, interpolation=cv2.INTER_AREA)
        
        if(psf.shape != (1, 1)):
            psf = psf / psf.sum()
        return psf, sampled_slope_deg, sampled_length

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
  
    def run(self, data):
        """
        Generates the crop pivot and call all the crop filters in self.filters
        Args:
            data: The data to be cropped from
        """
        assert len(data) == len(self.filters)
        # always ommit the last row and column in the coef data
        shape = np.asarray(data[0].shape[0:2])-1
        pivot = [ self.rng.randint(0, dim - self.size) for dim in shape ]
        pivot = np.asarray(pivot)
        size = np.asarray((self.size, self.size))
        
        crops = []
        for filter_crop, data_item in zip(self.filters, data):
            crop_data = filter_crop(data_item, pivot, size)
            crops.append(crop_data)
            
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
    
    def crop(self, data=None, pivot=None, size=None):
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
        
        cropped_data = data[p_y-size_y:p_y+size_y, p_x-size_x:p_x+size_x]
        return cropped_data
    
    def __call__(self, data=None, pivot=None, size=None):
        return self.crop(data=data, pivot=pivot, size=size)

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
    
    def crop(self, data=None, pivot=None, size=None):
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
        
        cropped_data = data[p_y:p_y+size_y, p_x:p_x+size_x]
        return cropped_data
    
    def __call__(self, data=None, pivot=None, size=None):
        return self.crop(data=data, pivot=pivot, size=size)

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
    
    def label(self, data):
        return {self.label_name: data}
    
    def __call__(self, data):
        return self.label(data)

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
        
    def mul(self, data):
        return data*self.val
    
    def __call__(self, data):
        return self.mul(data)

class Sub(object):
    """
    Subtract an value from data.
    """
    def __init__(self, val=0):
        self.val = val
        
    def sub(self, data):
        return data - self.val
    
    def __call__(self, data):
        return self.sub(data)