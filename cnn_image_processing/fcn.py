'''
Created on Jun 5, 2016

@author: isvoboda
'''
from __future__ import division

import caffe
import numpy as np
import multiprocessing
import time
import logging
import os

from filters.writers import ImageWriter
from cProfile import label

class FCN(multiprocessing.Process):
    """
    FCN feed forward the packet data through the FCN - fully convolutional
    network (CNN - convolutional network without fully (dense) connected
    layers). Compose the FCN packet data result an write it to png file. 
    """
    def __init__(self, in_queue=None, deploy=None, caffe_weights=None, 
                 in_blob=None, patch_size=None, batch_size=1, out_blob=None,
                 borders=None, path=None, caffe_mode=None, gpu_id=0):
        super(FCN, self).__init__()
        self.daemon = True  # Kill yourself if parent dies
        self.in_queue = in_queue
        self.deploy = deploy
        self.caffe_weights = caffe_weights
        self.in_blob = in_blob
        self.patch_size = np.asarray(patch_size)
        self.batch_size = batch_size
        self.out_blob = out_blob
        self.borders = np.asarray(borders) if borders != None else None
        self.writer = ImageWriter(d_path = path)
        self.caffe_mode = "GPU" if caffe_mode == None else caffe_mode
        self.gpu_id = gpu_id
#         self.timeout = 20
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
        
        orig_shape = np.asarray(fcn.blobs[self.in_blob].data.shape)
        scale_shape = self.patch_size[[2,0,1]] / orig_shape[1:]
        
        for in_key in fcn.inputs:
            orig_shape = np.asarray(fcn.blobs[in_key].shape)
            new_z, new_y, new_x = (orig_shape[1:] * scale_shape).astype(np.int)
            fcn.blobs[in_key].reshape(self.batch_size, new_z, new_y, new_x)
        
        if self.out_blob == None:
            self.out_blob = fcn.blobs.keys()[-1]  

        fcn.reshape()
        
        out_blob_shape = np.asarray(fcn.blobs[self.out_blob].shape)
        self.out_shape = out_blob_shape[1:]
        
        self.cnn_shape = self.patch_size[0:2] - 2*self.borders[0:2]
        self.out_scale = self.out_shape[1:3] / self.cnn_shape

        return fcn   
    
    def split(self, packet):
        """
        Split the data of packet into user defined size parts.
        Data is split into patches overlapped by the border reduced by the
        feed forwarding through the network. The right most column and bottom
        row are padded to be divideable by the FCN in_patch size * .
        """
        parts = []
        
        data = packet['data']
        label = packet['label']
        
        in_shape = self.patch_size[0:2]
        data_shape = np.asarray(data.shape[0:2])        
        
        div = data_shape / self.cnn_shape
        pad_size = np.ceil(div).astype(np.int)
        
        pad_borders = pad_size * self.cnn_shape - data_shape
        
        y_borders = (self.borders[0], self.borders[1]+pad_borders[0])
        x_borders = (self.borders[0], self.borders[1]+pad_borders[1])
        yxz_borders = [y_borders, x_borders, (0,0)]
        pad_data = np.pad(data, yxz_borders, mode='edge')
        
        part_id = 0
        n_parts = pad_size[0]*pad_size[1]
        packet['pad_parts_size'] = pad_size
        
        for y_crop in xrange( pad_size[0] ):
            for x_crop in xrange( pad_size[1] ):
                y_piv = y_crop * self.cnn_shape[0]
                x_piv = x_crop * self.cnn_shape[1]
                
                patch = pad_data[y_piv:y_piv+in_shape[0],
                                 x_piv:x_piv+in_shape[1], :]
                parts.append({'data': patch,
                              'pivot': np.asarray((y_crop, x_crop)),
                              'label': label,
                              'part_id': part_id,
                              'n_parts': n_parts })
                part_id += 1
        
        return parts
        
    def read(self, queue, in_packets, parts_meta):
        i_patch = 0
        i_batch = 0
        start_fetch = time.clock()  
        for packets in iter(queue.get, None):
            
            dest_shape = None
            orig_shape = None
            
            for packet in packets:
                label = packet['label']
                
                if label == 'shape':
                    dest_shape = packet['data'].shape
                
                else:
                    packet['parts'] = []
                    split_packet = self.split(packet)
                    parts_meta[label].extend(split_packet)
                    in_packets[label].append(packet)
                 
                    if 'orig_shape' in packet:
                        orig_shape = packet['orig_shape']
                    
                    if label == self.in_blob:
                        i_batch += len(split_packet)
                        i_patch += 1
            
            if dest_shape != None:
                dest_shape = np.asarray(dest_shape)
            elif orig_shape != None:
                dest_shape = np.asarray(orig_shape)
            else:
                data_shape = in_packets[self.in_blob][-1]['data'].shape[0:2]
                dest_shape = self.out_scale * np.asarray(data_shape)
            
            in_packets[self.in_blob][-1]['dest_shape'] = dest_shape 
            
            if i_batch >= self.batch_size:            
                break
        
        stop_fetch = time.clock()
        self.log.debug(" Fetch data time: {}".format(stop_fetch-start_fetch))
        self.log.debug(" Fetch packets: {}".format(i_patch))
        self.log.debug(" Packets split into: {} parts".format(i_batch))
        
        return i_batch >= self.batch_size
    
    def feed_fcn(self, fcn, parts_meta, batch_size):
        res = False
        for label in parts_meta.keys():
            if len(parts_meta[label]) >= batch_size:
                res = True
                for i_batch in xrange(batch_size):
                    part = parts_meta[label][i_batch]
                    data = part['data']
                    in_data = data.transpose([2,0,1]) #[z,y,x]
                    fcn.blobs[label].data[i_batch][...] = in_data
        
        return res
    
    def gather_parts(self, fcn, in_packets, parts_meta, batch_size):
        
        fcn_out_data = fcn.blobs[self.out_blob].data
        
        done_packets = []
        
        for i_batch in xrange(batch_size):
            part_data = np.copy(fcn_out_data[i_batch].transpose([1,2,0]))
            part_meta = parts_meta[self.in_blob].pop(0)
            pop_keys = parts_meta.viewkeys() - {self.in_blob}
            for key in pop_keys:
                if len(parts_meta[key]) > 0: parts_meta[key].pop(0)
            part_meta['data'] = part_data
            in_packets[self.in_blob][0]['parts'].append(part_meta)
        
            if part_meta['part_id'] == (part_meta['n_parts']-1):
                done_packets.append(in_packets[self.in_blob].pop(0))
                for key in pop_keys:
                    if len(in_packets[key]) > 0: in_packets[key].pop(0)
            
        return done_packets
            
    def build_img(self, packets):
        for packet in packets:
            part_shape = np.asarray(packet['parts'][0]['data'].shape)
            pad_parts_shape = packet['pad_parts_size']
            pad_data_shape = np.copy(part_shape)
            pad_data_shape[0:2] = part_shape[0:2] * pad_parts_shape
            img = np.zeros(pad_data_shape)
            
            for img_packet in packet['parts']:
                pivot = img_packet['pivot']
                y_ind, x_ind = (pivot*part_shape[0:2]).astype(np.int)
                img[y_ind:y_ind+part_shape[0],
                    x_ind:x_ind+part_shape[1]] = img_packet['data']
#             data_shape = packet['data'].shape
            
            packet['fcn_data'] = img[0:packet['dest_shape'][0],
                                     0:packet['dest_shape'][1]]
           
    def write_img(self, packets):
        while(len(packets) > 0):
            packet = packets.pop(0)
            img = packet['fcn_data']
            img_name = os.path.basename(packet['path']) 
            write_image_name = "{}_fcn.png".format(img_name)
            self.writer(write_image_name, img)
            self.log.info("Written: {}".format(write_image_name))

    def run(self):
        fcn = self.init_caffe()
        in_packets = {in_key: [] for in_key in fcn.inputs}
        parts_meta = {in_key: [] for in_key in fcn.inputs}
        
        flag_read = True
        
        while(flag_read):
            flag_read = self.read(self.in_queue, in_packets, parts_meta)
            
            batch_size = self.batch_size
            n_parts_meta = len(parts_meta[self.in_blob])
            if self.batch_size > n_parts_meta:
                batch_size = n_parts_meta
            
            while(batch_size > 0):
                start_feed = time.clock()  
                self.feed_fcn(fcn, parts_meta, batch_size)
                fcn.forward()
                stop_feed = time.clock()
                done_packets = self.gather_parts(fcn, in_packets, parts_meta,
                                                 batch_size)
                stop_gather = time.clock()
                self.build_img(done_packets)
                stop_compose = time.clock()
                self.write_img(done_packets)
                stop_write = time.clock()
                
                n_parts_meta = len(parts_meta[self.in_blob])
                if self.batch_size > n_parts_meta:
                    batch_size = n_parts_meta
                
                d_feed_t = stop_feed-start_feed
                d_read_t = stop_gather-stop_feed
                d_compose_t = stop_compose-stop_feed
                d_write_t = stop_write-stop_feed 
                self.log.debug(" Feed fnc time: {}ms".format(d_feed_t))
                self.log.debug(" FCN data read time: {}ms".format(d_read_t))
                self.log.debug(" Image compose time: {}ms".format(d_compose_t))
                self.log.debug(" Image write time: {}ms".format(d_write_t))
