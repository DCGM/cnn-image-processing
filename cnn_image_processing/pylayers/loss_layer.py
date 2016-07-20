'''
Created on May 27, 2016

@author: isvoboda
'''

from __future__ import print_function
from __future__ import division

from collections import namedtuple
import logging
import caffe
import numpy as np


class PyEuclideanLossL(caffe.Layer):

    """
    The Euclidian loss layer takes as input the 2 or 3 bottoms,
    bottom[0] is backward propagated.
    bottom[1] is the label

    Loss is defined as: (bottom[0] - bottom[end]) / [batch_size | n_pixels] / 2
    Parameters:
    -----------
      norm: string
        'batch_num', default normalize by number of samples in minibatch (w)
        'batch_size', normalize by size of minibatch (w * z * y * x)

    Layer prototxt definition:
    --------------------------
    layer {
      type: 'Python'
      name: 'Loss'
      top: 'loss'
      bottom: 'reconstruction'
      bottom: 'label'
      python_param {
        # the module name - the filename-that needs to be in $PYTHONPATH
        module: 'cnn_image_processing.pylayers'
        # the layer name - the class name in the module
        layer: 'PyEuclideanLossL'
        param_str: "norm: batch_num}"
      }
      # set loss weight so Caffe knows this is a loss layer.
      # since PythonLayer inherits directly from Layer,
      # this isn't automatically known to Caffe
      loss_weight: 1
    }
    """

    def setup(self, bottom, top):
        """
        Setup the layer
        """
        # check input pair
        if len(bottom) < 2:
            raise Exception("Need at least two inputs (data, label) \
            to compute distance.")

        self.log = logging.getLogger(__name__)

        self.dict_param = dict((key.strip(), val.strip()) for key, val in (
            item.split(':') for item in self.param_str.split(',')))

        if 'norm' in self.dict_param:
            self.norm = self.dict_param['norm']
            assert self.norm == 'batch_num' or self.norm == 'batch_size'
        else:
            self.norm = 'batch_num'

        self.log.info(self.param_str)

        # Compute the borders
        shape_cnn_y, shape__cnn_x = bottom[0].data.shape[2:]
        shape_y, shape_x = bottom[1].data.shape[2:]
        self.borders = (np.asarray((shape_y - shape_cnn_y,
                                    shape_x - shape__cnn_x))
                        // 2).astype(np.int)
        self.log.info("Bottom[0]  %s,  Bottom[1] %s border: %s",
                      bottom[0].data.shape, bottom[1].data.shape, self.borders)

        self.diff = np.zeros_like(bottom[0].data, dtype=np.float64)
        top[0].reshape(1)

    def reshape(self, bottom, top):
        """
        Reshape the activation - data blobs
        """
        # difference is shape of cnn data
        if self.diff.shape != bottom[0].data.shape:
            self.diff = np.zeros_like(bottom[0].data,
                                      dtype=np.float64)

            self.batch_size = bottom[0].num
            self.pixel_num = bottom[0].data.size

            # loss output is scalar
            top[0].reshape(1)

    def forward(self, bottom, top):
        """
        Feed forward
        """
        cropped_blob = self.crop(bottom[1], self.borders)

        self.diff[...] = (
            bottom[0].data - cropped_blob).astype(np.float64)

        # Euclidian loss
        if self.norm == 'batch_size':
            top[0].data[...] = np.sum(
                self.diff ** 2) / bottom[0].data.size / 2.
        elif self.norm == 'batch_num':
            top[0].data[...] = np.sum(self.diff ** 2) / bottom[0].num / 2.

    def backward(self, top, propagate_down, bottom):
        """
        Layer bacpropagation
        """
        for i in xrange(len(bottom)):
            if not propagate_down[i]:
                continue
            if i == 0:
                sign = 1  # label at bottom[1]: data - label
            else:
                sign = -1  # label at bottom[0]: label - data
            if self.norm == 'batch_size':
                bottom[i].diff[...] = sign * self.diff / bottom[i].data.size
            elif self.norm == 'batch_num':
                bottom[i].diff[...] = sign * self.diff / bottom[i].num

    def crop_all(self, bottom):
        """
        Crop all the bottoms according the self.borders
        """
        cropped = []
        for i_bot in xrange(len(bottom)):
            cropped.append(self.crop(bottom[i_bot], self.borders[i_bot]))
        return cropped

    def crop(self, blob, crop_size):
        """
        Crop the blob in y and x according the crop_size
        """
        x_border, y_border = crop_size
        crop_blob = blob.data[:, :,
                              x_border:-x_border or None,
                              y_border:-y_border or None]
        return crop_blob


class PyPSNRLossL(caffe.Layer):

    '''
    PSNR based loss layer

    To optimize in the descent, PSNR has to be mutiplied by -1
    (The higher PSNR the better data similarity but the idea is to descent -
    i.e flip the loss function by -1 * psnr and compute the gradient)

    Steps to derive the loss fce

    PSNR = 10 * log10(MAX^2/MSE);
    MAX = 1
    MSE = 1/(M*N) * Sum(Sum( (x_{m,n} - l_{m,n})^2 ))

    Differentiation according every data pixel x_{m,n}:

    d -1*10 * log_10(MAX/MSE) / dx_{m,n} = 20*(x-l)/(MSE * MSE.size)
    i.e. d/dx =  20/log(10) * 1/diff

    '''

    def __init__(self, arg):
        '''
        Constructor initilize the layer
        '''
        super(PyPSNRLossL, self).__init__(arg)
        self.log = logging.getLogger(__name__ + type(self).__name__)
        self.log_const = -20 / np.log(10)
        self.max = 1
        self.eps = (self.max / 128) ** 2
        self.diff = None

    PSNR_tuple = namedtuple('PSNR_tuple', ['psnr', 'ssd', 'mse', 'diff'])
    '''
    Return type of psnr method
    '''

    def setup(self, bottom, top):
        '''
        Caffe base setup
        '''
        if len(bottom) != 2:
            raise Exception("Need two inputs to compute distance.")
        self.diff = np.zeros_like(bottom[0].data, dtype=np.float32)

    def reshape(self, bottom, top):
        '''
        Reshape the layer data blobs if needed
        '''
        # check input dimensions match
        if bottom[0].count != bottom[1].count:
            raise Exception("Inputs must have the same dimension.")
        # difference is shape of inputs
        self.diff = np.zeros_like(bottom[0].data, dtype=np.float32)
        # loss output is scalar
        top[0].reshape(1)

    def forward(self, bottom, top):
        '''
        Feed forward - computes the Loss and related gradients
        '''
        psnr_list = []
        for i_batch in xrange(bottom[0].data.shape[0]):
            psnr_tuple = self.psnr(bottom[0].data[i_batch],
                                   bottom[1].data[i_batch])
            psnr_list.append(psnr_tuple.psnr)

            mse_eps = (psnr_tuple.mse + self.eps)
            diff = -1 * self.log_const / (mse_eps * psnr_tuple.diff.size)
            self.diff[i_batch, ...] = diff * psnr_tuple.diff

        top[0].data[...] = -1 * np.average(psnr_list)

    def backward(self, top, propagate_down, bottom):
        '''
        Backpropagation
        '''
        bottom[0].diff[...] = self.diff / bottom[0].num

    def psnr(self, data, label):
        """
        Compute PSNR
        """
        self.
        diff = (data - label).astype(dtype=np.float64)
        ssd = (diff ** 2).sum()
        mse = ssd / float(diff.size)
        psnr = 10 * np.log10(self.max ** 2 / mse)
        return self.PSNR_tuple(psnr, ssd, mse, diff)
