from __future__ import print_function
from __future__ import division
import sys
import cv2
import numpy as np
import caffe
import yaml
from utils import RADequeue
import logging

module_logger = logging.getLogger(__name__)
module_logger.setLevel(logging.INFO)

class PySubL(caffe.Layer):
    """A layer that compute bottom[0].data - bottom[1].data"""
    
    def setup(self, bottom, top):
        if len(bottom) != 2 :
            raise Exception("PySubL has to have 2 inputs.")
        
        self.i_data = 0
        self.i_label = 1
        # Compute the border
        label_shape = np.asarray(bottom[self.i_label].data.shape[2:])
        data_shape = np.asarray(bottom[self.i_data].data.shape[2:])
        self.borders = label_shape - data_shape
        #https://www.python.org/dev/peps/pep-0238/ ie floor division
        self.borders //= 2
        
    def reshape(self, bottom, top):
        top[0].reshape(*bottom[self.i_data].data.shape)
     
    def forward(self, bottom, top):
        (i_crop_x, i_crop_y) = self.borders
        len_x = bottom[self.i_data].data.shape[2]
        len_y = bottom[self.i_data].data.shape[3]
        label_data = bottom[self.i_label].data
        crop_data = label_data[:, :, i_crop_x:i_crop_x + len_x,
                               i_crop_y:i_crop_y + len_y]
         
        top[0].data[...] = crop_data - bottom[self.i_data].data
     
    def backward(self, top, propagate_down, bottom):
        bottom[self.i_data].diff[...] = top[0].diff
        
class PyAddL(caffe.Layer):
    """A layer that compute bottom[0].data + bottom[1].data"""
    
    def setup(self, bottom, top):
        if len(bottom) != 2 :
            raise Exception("PySubL has to have 2 inputs.")
        self.i_data = 0
        self.i_label = 1
        # Compute the border
        label_shape = np.asarray(bottom[self.i_label].data.shape[2:])
        data_shape = np.asarray(bottom[self.i_data].data.shape[2:])
        self.borders = label_shape - data_shape
        self.borders //= 2
        
    def reshape(self, bottom, top):
        top[0].reshape(*bottom[self.i_data].data.shape)
    
    def forward(self, bottom, top):
        (i_crop_x, i_crop_y) = self.borders
        len_x = bottom[self.i_data].data.shape[2]
        len_y = bottom[self.i_data].data.shape[3]
        label_data = bottom[self.i_label].data
        crop_data = label_data[:, :, i_crop_x:i_crop_x + len_x,
                               i_crop_y:i_crop_y + len_y]
        top[0].data[...] = crop_data + bottom[self.i_data].data
    
    def backward(self, top, propagate_down, bottom):
        bottom[self.i_data].diff[...] = top[0].diff
        
class PyCropL(caffe.Layer):
    """
    Crop the center of bottom[0].data according the bottom[1].data
    """
    def setup(self, bottom, top):
        if len(bottom) != 2 :
            raise Exception("PyCropL has to have 2 inputs.")
        self.i_data = 0
        self.i_label = 1
        # Compute the border
        label_shape = np.asarray(bottom[self.i_label].data.shape[2:])
        data_shape = np.asarray(bottom[self.i_data].data.shape[2:])
        self.borders = data_shape - label_shape
        if not np.all(self.borders >= 0):
            raise Exception("Bottom input 0 is smaller then the"
            " crop refference 1.")
        self.borders //= 2
        
    def reshape(self, bottom, top):
        top[0].reshape(*bottom[self.i_label].data.shape)
    
    def forward(self, bottom, top):
        (i_crop_x, i_crop_y) = self.borders
        len_x = bottom[self.i_label].data.shape[2]
        len_y = bottom[self.i_label].data.shape[3]
        data = bottom[self.i_data].data
        crop_data = data[:, :, i_crop_x:i_crop_x + len_x,
                               i_crop_y:i_crop_y + len_y]
        top[0].data[...] = crop_data
    
    def backward(self, top, propagate_down, bottom):
        """
        ToDo realy pass only the i_data?
        """
        pad_params=((0,0),(0,0),
                    (self.borders,self.borders),(self.borders,self.borders))
        bottom[self.i_data].diff[...] = np.pad(top[0].diff, pad_params,
                                               mode='constant',
                                               constant_values=(0,0))

class PyPSNRL(caffe.Layer):
    """
    Compute a PSNR of two inputs and PSNR with IPSNR of 3 inputs
    Args:
        max: float
            The maximum value in the input data used to compute the PSNR.
            Default is 255
	history_size: uint
	     Size of history a floating avarge is computed from.
    """
    def setup(self, bottom, top):
        self.log = logging.getLogger(__name__ + ".PyPSNRL")
        if len(bottom) < 2 or len(bottom) > 3:
            raise Exception("Need two inputs at least or 3 at most.")
        self.dict_param = dict((key.strip(), val.strip()) for key, val in
                               (item.split(':') for item in
                                self.param_str.split(',')))
        if 'max' in self.dict_param:
            self.max = float(self.dict_param['max'])
        else:
            self.max = 255
        if 'history_size' in self.dict_param:
            self.history_size = np.uint(self.dict_param['history_size'])
        else:
            self.history_size = 50
            
        self.psnr_buffer = RADequeue(max_size=self.history_size)
       
    def reshape(self, bottom, top):
        if len(top) > len(bottom):
            raise Exception("Layer produce more outputs then has its inputs.")
        
        for i_input in xrange(len(top)):
            top[i_input].reshape(*bottom[i_input].data.shape)
    
    def forward(self, bottom, top):
        psnr = self.psnr(bottom)
        msg = "\n".join('PSNR {}: {}'.format(*val) for val in enumerate(psnr))
        self.log.info(msg)
        if len(psnr) == 2:
            self.log("iPSNR: {}".format(psnr[0] - psnr[1]))
    
    def backward(self, top, propagate_down, bottom):
        for i_diff in xrange(len(top)):
            bottom[i_diff].diff[...] = top[i_diff].diff
    
    def psnr(self, bottom):
        results = []
        for i_input in xrange(len(bottom)-1):
            diff = bottom[-1].data - bottom[i_input].data
            ssd = (diff**2).sum()
            mse = ssd / float(diff.size)
            if mse <= 0:
                results.append(np.nan)
            else:
                psnr = 10 * np.log10(self.max**2 / mse)
                results.append(psnr)
        return results

class PyEuclideanLossLayer(caffe.Layer):
    """The Euclidian loss layer takes as input the cnn output data,
    label, and optional cnn input data. If cnn input data is defined
    the layer prints the iPSNR of cnn input data and the
    cnn output reconstruction.
    
    Loss is defined as: (cnn output data - label) / [batch_size | n_pixels]
    Parameters:
    -----------
      pixel_norm: boolean
        True computes loss normalized by pixels,
        False loss normalized by batch size
      
      vis: boolean
        True visualize the layer input, False do not visualize
      
      vis_scale: float
        scale factor of visualized data
      
      vis_normalize: float
        value normalization factor
        
      vis_mean: float
        mean added to the visualized data
      
      print: boolean
        print the PSNR & iPSNR
      
      print_iter: int
        print PSNR and iPSNR every ith iteration
      
      
        
    Layer prototxt definition:
    --------------------------
    layer {
      type: 'Python'
      name: 'loss'
      top: 'loss'
      bottom: 'reconstruction'
      bottom: 'label'
      bottom: 'data'
      python_param {
        # the module name - the filename-that needs to be in $PYTHONPATH
        module: 'pylayers'
        # the layer name - the class name in the module
        layer: 'PyEuclideanLossLayer'
        param_str: "loss_opt: {'pixel_norm': True, 'vis': True, 'vis_scale': 6,
        'vis_mean': 127, 'vis_normalize': 0.004, 'print': True,
        'print_iter': 50}"
      }
      # set loss weight so Caffe knows this is a loss layer.
      # since PythonLayer inherits directly from Layer,
      # this isn't automatically known to Caffe
      loss_weight: 1
    }
    """
    
    def setup(self, bottom, top):
        # check input pair
        if len(bottom) < 2:
            raise Exception("Need at least two inputs (data, label) \
            to compute distance.")
        
        yaml_opt = yaml.load(self.param_str)
        
        if 'vis' in yaml_opt['loss_opt']:
            self.vis = yaml_opt["loss_opt"]['vis']
        else:
            self.vis = False
        
        if 'vis_scale' in yaml_opt['loss_opt']:
            self.vis_scale = yaml_opt["loss_opt"]['vis_scale']
        else:
            self.vis_scale = 1.
        
        if 'vis_normalize' in yaml_opt['loss_opt']:
            self.normalize = yaml_opt["loss_opt"]['vis_normalize']
        else:
            self.normalize = 1.
        
        if 'vis_mean' in yaml_opt['loss_opt']:
            self.vis_mean = yaml_opt["loss_opt"]['vis_mean']
        else:
            self.vis_mean = 0
        
        if 'pixel_norm' in yaml_opt['loss_opt']:
            self.pixel_norm = yaml_opt["loss_opt"]['pixel_norm']
        else:
            self.pixel_norm = False
          
        if 'print' in yaml_opt['loss_opt']:
            self.b_print = yaml_opt["loss_opt"]['print']
        else:
            self.b_print = False
          
        if 'print_iter' in yaml_opt['loss_opt']:
            self.print_iter = yaml_opt["loss_opt"]['print_iter']
        else:
            self.print_iter = 1
        
        
        print("PyEuclidianLossLayer vis: {}, vis_scale: {}, vis_normalize: {},"
        " vis_mean: {}, pixel_norm: {}, print: {}, print_iter: {}"\
              .format(self.vis, self.vis_scale, self.normalize, self.vis_mean,
                      self.pixel_norm, self.b_print, self.print_iter))
        
        self.iteration = np.uint(0)
        self.step = 50
        self.historySize = 200
        self.PSNR = []
        self.IPSNR = []
        
        self.i_cnn_data = 0
        self.i_label = 1
        self.i_data = 2
        # Compute the label cnn_data border
        (x_label, y_label) = bottom[self.i_label].data.shape[2:]
        (x_cnn_data, y_cnn_data) = bottom[self.i_cnn_data].data.shape[2:]
        self.label_cnn_data_borders =\
          np.asarray([x_label, y_label]) - np.asarray([x_cnn_data, y_cnn_data])
        self.label_cnn_data_borders //= 2
        
        print("PyEuclidianLossLayer label {} cnn output {} border: {}"\
              .format(bottom[self.i_label].data.shape,
                      bottom[self.i_cnn_data].data.shape,
                      self.label_cnn_data_borders))
        if(len(bottom) > 2):
            # Compute the data label
            (x_data, y_data) = bottom[self.i_data].data.shape[2:]
            self.data_cnn_data_borders =\
            np.asarray([x_data, y_data]) - np.asarray([x_cnn_data, y_cnn_data])
            self.data_cnn_data_borders //= 2
            print("PyEuclidianLossLayer data {} cnn output {} border: {}"
                  .format(bottom[self.i_data].data.shape,
                          bottom[self.i_cnn_data].data.shape,
                          self.data_cnn_data_borders))
            
        self.diff = np.zeros_like(bottom[self.i_cnn_data].data,
                                      dtype=np.float64)
        top[0].reshape(1)
#         self.reshape_count = 0
        self.batch_size =  bottom[self.i_cnn_data].num
        self.pixel_num = bottom[self.i_cnn_data].data.size

    def reshape(self, bottom, top):        
#         difference is shape of cnn data
        if self.diff.shape != bottom[self.i_cnn_data].data.shape:
            self.diff = np.zeros_like(bottom[self.i_cnn_data].data,
                                      dtype=np.float64)
            
            self.batch_size =  bottom[self.i_cnn_data].num
            self.pixel_num = bottom[self.i_cnn_data].data.size
            
            # loss output is scalar
            top[0].reshape(1)

    def forward(self, bottom, top):    
        crop_label = self.crop( bottom[self.i_label],
                                self.label_cnn_data_borders)

        self.diff[...] = bottom[self.i_cnn_data].data - crop_label
        
        PSNR, SSD = self.psnr(self.diff)
        self.appendPSNR(PSNR, self.PSNR)
        #Euclidian loss
        if self.pixel_norm:
            top[0].data[...] = SSD / self.pixel_num / 2.
        else:
            top[0].data[...] = SSD / self.batch_size / 2.
        
        
        if self.b_print and self.iteration % self.print_iter == 0:
            print("Forward Loss: {}".format(np.squeeze(top[0].data)),
                file=sys.stderr) 
            print("PSNR: {}".format(np.average(self.PSNR)), file=sys.stderr) 
            if len(bottom) == 3:
                crop_data = self.crop(bottom[self.i_data],
                                      self.data_cnn_data_borders)
                diff_data = crop_data - crop_label
                PSNR_data, SSD_data = self.psnr(diff_data)
                self.appendPSNR(PSNR - PSNR_data, self.IPSNR)
                print("iPSNR: {}".format(np.average(self.IPSNR)),
                      file=sys.stderr)
        
        if self.vis and self.iteration % self.step == 0:    
            vis_param = {"Label": crop_label[0],
                         "CNN data": bottom[self.i_cnn_data].data[0]}
            if len(bottom) >= 3:
                vis_param["Data"] = bottom[self.i_data].data[0]
          
            self.visualize(**vis_param)
        
        self.iteration += 1

    def backward(self, top, propagate_down, bottom):
        for i in xrange(len(bottom)):
            if not propagate_down[i]:
                continue
            if i == 0:
                sign = 1
            else:
                sign = -1
            if self.pixel_norm:
                bottom[i].diff[...] = sign * self.diff / bottom[i].data.size
            else:
                bottom[i].diff[...] = sign * self.diff / bottom[i].num
    
    def crop(self, blob, crop_size):
        (x_border, y_border) = crop_size
        crop_blob = blob.data[:, :, x_border:-x_border or None,
                              y_border:-y_border or None]
        return crop_blob
  
    def psnr(self, diff):
        SSD = np.sum(diff**2)
        MSE = SSD / float(diff.size)
        max_i = 255.*self.normalize
        PSNR = 10 * np.log10( max_i**2 / MSE)
        return PSNR, SSD

    def appendPSNR(self, PSNR, psnr_buf):
        psnr_buf.append(PSNR)
        if len(psnr_buf) > self.historySize:
            psnr_buf = psnr_buf[1:]
      
    def visualize(self, **kwargs):
#         img_pair = []
        for key, value in kwargs.items():
            img = value.transpose(1, 2, 0) / self.normalize
            img += self.vis_mean
#             img_pair.append(img)
            preview_resized = cv2.resize(img, (0, 0), fx=self.vis_scale,
                                         fy=self.vis_scale)
            
            cv2.imshow(key, preview_resized/255.)
#         cv2.imshow(key, np.vstack(img_pair)/255.)
        cv2.waitKey(5)     

class CroppedEuclideanLossVisLayer(caffe.Layer):
    """
    Compute the Euclidean Loss on aligned image in the same manner as the C++ EuclideanLossLayer.
    """

    def setup(self, bottom, top):
        # check input pair
        if len(bottom) < 2:
            raise Exception("Needs two inputs to compute distance. Possibly more to show.")

        self.historySize = 200
        self.PSNR = []
        self.IPSNR = []

        self.options = dict([tuple(s.split(':')) for s in self.param_str.split()])

    def crop(self, blob):
        crop = blob
        b2 = (crop.shape[2] - self.shape[2]) // 2
        if b2 > 0:
            crop = crop[:, :, b2:-b2, :]
        b3 = (crop.shape[3] - self.shape[3]) // 2
        if b3 > 0:
            crop = crop[:, :, :, b3:-b3]
        return crop

    def reshape(self, bottom, top):

        # check input dimensions match
        self.shape = [x for x in bottom[0].data.shape]
        for bID in range(1, len(bottom)):
            if self.shape[0] != bottom[bID].shape[0]:
                raise Exception("Number of images in bottom blobs must "\
                                " match. Got %d and %d" % \
                                 (self.shape[0], bottom[bID].shape[0]))
            if self.shape[1] != bottom[bID].shape[1]:
                raise Exception("Number of channels in bottom blobs "\
                                " must match. Got %d and %d" %\
                                (self.shape[1], bottom[bID].shape[1]))

            self.shape[2] = min(self.shape[2], bottom[bID].shape[2])
            self.shape[3] = min(self.shape[3], bottom[bID].shape[3])

        # size of difference is the same as of the first input
        self.diff = np.zeros(self.shape, dtype=np.float32)
        # loss output is scalar
        top[0].reshape(1)


    def computePSNR(self, diff):
        SSD = np.sum(diff ** 2)
        MSE = SSD / diff.size
        PSNR = 10 * np.log10(255. / MSE)
        return PSNR, SSD


    def forward(self, bottom, top):
        if 'show' in self.options:
            for bID in range(len(bottom)):
                img = self.crop(bottom[bID].data)[0, :, :, :]
                img = img.transpose([1, 2, 0])
                cv2.imshow('bottom%d' % bID,
                           cv2.resize(img + 0.5, (0, 0), fx=3, fy=3))
            cv2.waitKey(int(self.options['show']))

        # crop bottom[1] to match bottom[0]
        croppedTruth = self.crop(bottom[1].data)

        self.diff = (bottom[0].data - croppedTruth).astype(np.float64)
        PSNR, SSD = self.computePSNR(self.diff)
        top[0].data[...] = SSD / bottom[0].num / 2.

        self.PSNR.append(PSNR)
        if len(self.PSNR) > self.historySize:
            self.PSNR = self.PSNR[1:]
        print("PSNR: {}".format(reduce(lambda x, y: x + y, self.PSNR) /
        len(self.PSNR)), file=sys.stderr)

        if len(bottom) >= 3:
            croppedReference = self.crop(bottom[2].data)
            diffRef = (croppedTruth - croppedReference).astype(np.float64)
            PSNR2, SSD = self.computePSNR(diffRef)
            self.IPSNR.append(PSNR - PSNR2)
            if len(self.IPSNR) > self.historySize:
                self.IPSNR = self.IPSNR[1:]
            print("IPSNR: {}".format(reduce(lambda x, y: x + y, self.IPSNR) / 
            len(self.IPSNR), file=sys.stderr))


    def backward(self, top, propagate_down, bottom):
        i = 0
        if not propagate_down[i]:
            return
        if i == 0:
            sign = 1
        else:
            sign = -1
        bottom[i].diff[...] = sign * self.diff / bottom[i].num
