'''
Created on Jun 5, 2016

@author: isvoboda
'''
import caffe
import numpy as np
import multiprocessing
import time
import logging

class FCN(multiprocessing.Process):
    """
    FCN feed forward the data and through the FCN - fully convolutional
    network (CNN - convolutional network without fully (dense) connected
    layers).
    """
    def __init__(self, train_in_queue=None, deploy=None, caffe_weights=None, 
                 input_blob=None, patch_size=None, batch_size=1,
                 caffe_mode=None, gpu_id=0):
        super(FCN, self).__init__()
        self.daemon = True  # Kill yourself if parent dies
        self.in_queue = train_in_queue
        self.deploy = deploy
        self.caffe_weights = caffe_weights
        self.input_blob = input_blob
        self.patch_size = np.asarray(patch_size)
        self.batch_size = batch_size
        self.caffe_mode = "GPU" if caffe_mode == None else caffe_mode
        self.gpu_id = gpu_id
        self.log = logging.getLogger(__name__ + ".FCN")
        
    def init_caffe(self):
        """
        Initialize the caffe FCN.
        """
        if(self.caffe_mode.lower() == "gpu"):
            caffe.set_mode_gpu()
            caffe.set_device(self.gpu_id)
        else:
            caffe.set_mode_cpu()
        
        self.log.info(" deploy: {}".format(self.deploy))
        self.log.info(" weights: {}".format(self.caffe_weights))
        self.log.info(" input size: {}".format(self.patch_size))
        self.log.info(" batch size: {}".format(self.batch_size))
        
        fcn = caffe.Net(self.deploy, self.caffe_weights, caffe.TEST)
        fcn.blobs[self.input_blob].reshape(self.batch_size,
                                           -1,
                                           *self.patch_size)         
        fcn.reshape()

        

        return fcn   
    
    def spatial_yx_borders(self, fcn):
        diff = np.asarray(fcn.blobs[0].shape) - np.asarray(fcn.blobs[-1].shape) 
        return diff[2:]
    
    def edge_spatial_yx_borders(self, input_data, borders):
        yxz_borders = [(borders[0],borders[0]), (borders[0],borders[0]), (0,0)]
        return np.pad(input_data, yxz_borders, mode='edge')

    def split(self, input_data):
        parts = []
        steps = np.asarray(input_data.shape)[0:2] // self.patch_size
        y_size, x_size = self.patch_size
        
        for y_step in xrange(0,steps[0]):
            for x_step in xrange(0,steps[1]):
                yind = y_step * self.patch_size[0]
                xind = x_step * self.patch_size[1]
        
                part = input_data[yind:yind+y_size, xind:xind+x_size, :]
                parts.append(part)
    
    def run(self):
        
        fcn = self.init_caffe()     
        i_batch = 0
        i_iter = 0
        start_fetch = time.clock()  
        for packet in iter(self.in_queue.get, None):
            if i_batch < self.batch_size:
                for packet_item in packet:
                    key, packet_data = packet_item.items()[0]
                    packet_data = packet_data.transpose([2, 0, 1]) # [z y x]
                    fcn.blobs[key].data[i_batch][...] = packet_data
                i_batch += 1
                continue
            stop_fetch = time.clock()
               
            start_forward = time.clock()
            fcn.forward_all()
            stop_forward = time.clock()
            
            self.log.debug("Forward time: {}"
                           .format(stop_forward-start_forward))
            self.log.debug("Fetch data time: {}"
                           .format(stop_fetch-start_fetch))
            self.log.debug("Iteration time: {}"
                           .format(stop_forward-start_fetch))
            self.log.debug("Iteration: {}".format(i_iter))
      
            start_fetch = time.clock()
                  
            i_batch = 0
