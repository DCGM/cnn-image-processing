'''
Created on May 27, 2016

@author: isvoboda
'''

from __future__ import print_function
from __future__ import division

import caffe
import cv2
import numpy as np
import logging
# pylint: disable=import-error,no-name-in-module
from distutils.util import strtobool

from ..utils import RoundBuffer


class PyEuclideanLossLayer(caffe.Layer):
    """The Euclidian loss layer takes as input the 2 or 3 bottoms,
    bottom[0] is backward propagated.
    bottom[1] except bottom[1] == last bottom is used to compute the iPSNR.
    iPSNR is computed as PSNR(bottom[0]) - PSNR(bottom[1])
    bottom[-1] is the label according is computed the PSNR

    Loss is defined as: (bottom[0] - bottom[end]) / [batch_size | n_pixels] / 2
    Parameters:
    -----------
      pixel_norm: boolean
        True computes loss normalized by number of pixels,
        False loss normalized by batch size

      psnr_max: float
          Value of the maximum value the PSNR is computed from
          For uint image it is 255
          For normalized float image it is 1

      vis: boolean
        True visualize the layer input, False do not visualize

      vis_scale: float
        scale factor of visualized data

      vis_normalize: float
        value normalization factor
        default: 1

      vis_mean: float
        mean added to the visualized data
        default: 0

      print: boolean
        print the PSNR & iPSNR (if 3 inpouts available)

      print_iter: int
        print PSNR (and iPSNR) every ith iteration



    Layer prototxt definition:
    --------------------------
    layer {
      type: 'Python'
      name: 'Loss'
      top: 'loss'
      bottom: 'reconstruction'
      bottom: 'label'
      bottom: 'data'
      python_param {
        # the module name - the filename-that needs to be in $PYTHONPATH
        module: 'cnn_image_processing'
        # the layer name - the class name in the module
        layer: 'PyEuclideanLossLayer'
        param_str: "pixel_norm: True, psnr_max: 255, vis: True, vis_scale: 6,
        vis_mean: 127, vis_normalize: 0.004, print: True, print_iter: 50}"
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

        if 'vis' in self.dict_param:
            self.vis = strtobool(self.dict_param['vis'])
        else:
            self.vis = False

        if 'vis_scale' in self.dict_param:
            self.vis_scale = float(self.dict_param['vis_scale'])
        else:
            self.vis_scale = 1.

        if 'vis_normalize' in self.dict_param:
            self.normalize = float(self.dict_param['vis_normalize'])
        else:
            self.normalize = 1.

        if 'vis_mean' in self.dict_param:
            self.vis_mean = float(self.dict_param['vis_mean'])
        else:
            self.vis_mean = 0

        if 'pixel_norm' in self.dict_param:
            self.pixel_norm = strtobool(self.dict_param['pixel_norm'])
        else:
            self.pixel_norm = False

        if 'print' in self.dict_param:
            self.b_print = strtobool(self.dict_param['print'])
        else:
            self.b_print = False

        if 'print_iter' in self.dict_param:
            self.print_iter = int(self.dict_param['print_iter'])
        else:
            self.print_iter = 1

        if 'psnr_max' in self.dict_param:
            self.max = float(self.dict_param['psnr_max'])
        else:
            self.max = 255

        self.log.info(self.param_str)

        self.iteration = np.uint(0)
        self.step = 5
        self.history_size = 200
        self.psnr_buffers = [RoundBuffer(max_size=self.history_size)
                             for _ in xrange(len(bottom) - 1)]

        # Compute the borders
        self.borders = []
        (shape_cnn_y, shape__cnn_x) = bottom[0].data.shape[2:]
        for i_bot in xrange(len(bottom)):
            (shape_y, shape_x) = bottom[i_bot].data.shape[2:]
            self.borders.append(np.asarray((shape_y - shape_cnn_y,
                                            shape_x - shape__cnn_x)) // 2)
            self.log.info("Bottom[0]  {},  Bottom[{}] {} border: {}"
                          .format(bottom[0].data.shape,
                                  i_bot,
                                  bottom[i_bot].data.shape,
                                  self.borders[i_bot]))

        self.diff = np.zeros_like(bottom[0].data, dtype=np.float64)
        top[0].reshape(1)

        self.batch_size = bottom[0].num
        self.pixel_num = bottom[0].data.size

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
        cropped_blobs = self.crop_all(bottom)

        l_psnr, l_diff, l_ssd = self.psnr(cropped_blobs)
        self.diff[...] = l_diff[0]

        for i_val, val in enumerate(l_psnr):
            self.psnr_buffers[i_val].append_round(val)

        # Euclidian loss
        if self.pixel_norm:
            top[0].data[...] = l_ssd[0] / self.pixel_num / 2.
        else:
            top[0].data[...] = l_ssd[0] / self.batch_size / 2.

        if self.b_print and self.iteration % self.print_iter == 0:
            self.log.info(" Forward Loss: {}".format(np.squeeze(top[0].data)))

            avg_psnr = [sum(val) / val.size for val in self.psnr_buffers]

            avg_msg = " PSNR average of {} samples".format(
                self.psnr_buffers[0].size)
            self.log.info(avg_msg)

            msg = " ".join(' PSNR bottom[{}]: {}'
                           .format(*val) for val in enumerate(avg_psnr))

            self.log.info(msg)
            if len(l_psnr) == 2:
                self.log.info(" iPSNR: {}".format(avg_psnr[0] - avg_psnr[1]))

        if self.vis and self.iteration % self.step == 0:
            vis_param = {}
            for i_bot in xrange(len(bottom)):
                vis_param["bottom[{}]".format(i_bot)] = cropped_blobs[i_bot][0]

            self.visualize(**vis_param)

        self.iteration += 1

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
            if self.pixel_norm:
                bottom[i].diff[...] = sign * self.diff / bottom[i].data.size
            else:
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
        (x_border, y_border) = crop_size
        crop_blob = blob.data[
            :, :, x_border:-x_border or None, y_border:-y_border or None]
        return crop_blob

    def psnr(self, blobs):
        """
        Compute PSNR
        """
        results = []
        diffs = []
        ssds = []
        for i_input in xrange(len(blobs) - 1):
            diff = (blobs[i_input] - blobs[-1])\
                .astype(dtype=np.float64)
            ssd = (diff**2).sum()
            mse = ssd / float(diff.size)
            if mse == 0:
                results.append(np.nan)
            else:
                psnr = 10 * np.log10(self.max**2 / mse)
                results.append(psnr)
                diffs.append(diff)
                ssds.append(ssd)
        return results, diffs, ssds

    def visualize(self, **kwargs):
        """
        Visualize the input data - Depricated use VisLayer
        """
        #         img_pair = []
        for key, value in kwargs.items():
            img = value.transpose(1, 2, 0) / self.normalize
            img += self.vis_mean
#             img_pair.append(img)
            preview_resized = cv2.resize(img, (0, 0), fx=self.vis_scale,
                                         fy=self.vis_scale)

            cv2.imshow(key, preview_resized / 255.)
#         cv2.imshow(key, np.vstack(img_pair)/255.)
        cv2.waitKey(5)
