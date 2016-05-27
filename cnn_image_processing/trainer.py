'''
Created on May 27, 2016

@author: isvoboda
'''

import Queue
import multiprocessing
import caffe
import logging
import collections
import numpy as np
import cv2
import time

class Trainer(multiprocessing.Process):
    """
    Trainer call the caffe solver - train and prepare the input
    cnn input batches.
    """
    def __init__(self, in_queue=None, solver_file=None, max_iter=100,
                 batch_size=64, buffer_size=256, caffe_weights=None,
                 caffe_solverstate=None, caffe_mode=None, gpu_id=0):
        """
        Trainer constructor
        Args:
          in_queue: The queue the data are read from.
          solver: initialized Caffe solver.
          max_iter: number of train iterations.
          batch_size: Size of the train batch data.
          buffer_size: Size of the internal buffer - used with threads.
          caffe_weights: Weights to load.
          caffe_solverstate: Solverstate to restore.
          caffe_mode: Set the CPU or GPU caffe mode.
          gpu_id: The gpu id on a multi gpu system.
        """
        super(Trainer, self).__init__()
        self.daemon = True  # Kill yourself if parent dies
        self.in_queue = in_queue
        self.solver_file = solver_file
        self.max_iter = max_iter
        self.batch_size = batch_size
        self.buffer = Queue.Queue(buffer_size)
        self.caffe_weights = caffe_weights
        self.caffe_solverstate = caffe_solverstate
        self.caffe_mode = "GPU" if caffe_mode == None else caffe_mode
        self.gpu_id = gpu_id
        self.stat = None
        self.log = logging.getLogger(__name__ + ".Trainer")

    def center_initialization(self, net, step=1):
        """
        Center the net parameters - subtract the non-zero mean
        """
        for l in net.params:
            if len(net.params[l][0].data.shape) == 4:
                params = net.params[l][0]
                average = np.average(params.data, (1,2,3))
                average = average.reshape( (-1, 1, 1, 1))
                params.data[...] = params.data - (average*step)

    def init_caffe(self, solver_file, data_shapes):
        """
        Initialize the caffe solver.
        """
        if(self.caffe_mode.lower() == "gpu"):
            caffe.set_mode_gpu()
            caffe.set_device(self.gpu_id)
        else:
            caffe.set_mode_cpu()
        print("SOLVER FILE: {}".format(solver_file))
        solver = caffe.SGDSolver(solver_file)
        
        if self.caffe_solverstate is not None:
            solver.restore(self.caffe_solverstate)
        elif self.caffe_weights is not None:
            solver.net.copy_from(self.caffe_weights)
        
        for key, shape in data_shapes.items():
            solver.net.blobs[key].reshape(*shape)
            
        solver.net.reshape()
        for test_net in solver.test_nets:
            for key, shape in data_shapes.items():
                test_net.blobs[key].reshape(*shape)
            test_net.reshape()

        self.log.info("Net - centering the parameters.")
        self.center_initialization(solver.net)
        self.stat = ActivationStat(solver.net)
        return solver

    def thr_fetch(self, in_buffer, data_shape, label_shape):
        """
        Fetch the mini-batch data into the buffer.
        ToDo - use the proper labels from the packet  
        """
        i_batch = 0
        nd_data = np.ndarray(shape=data_shape, dtype=np.int64)
        nd_label = np.ndarray(shape=label_shape, dtype=np.uint8)
        fetch_start = time.clock()
        for packet in iter(self.in_queue.get, None):
            if i_batch < self.batch_size:
                nd_data[i_batch][...] = packet[0].transpose([2, 0, 1])  # [channels, y, x]
                nd_label[i_batch][...] = packet[1].transpose([2, 0, 1]) # [channels, y, x]
                i_batch += 1
                continue
            
            fetch_stop = time.clock()
            # Data have to be copied otherwise are modified by this thread
            self.buffer.put({'data_batch':np.array(nd_data, copy=True),
                             'label_batch': np.array(nd_label, copy=True)})
            self.log.debug("Thread fetch time of batch: {}".
                          format(fetch_stop-fetch_start))
            i_batch = 0
            fetch_start = time.clock()

    def run(self):
        # Get the first packet in the stream to find out the proper shapes
        batch_n = self.batch_size
        in_packet = self.in_queue.get(block=True, timeout=None)
        dict_shapes = {}
        for packet in in_packet:
            label, data = packet.items()[0]
            dim = data.shape
            blob_shape = [batch_n, dim[2], dim[0], dim[1]]
            dict_shapes[label] = blob_shape    

        # Initialize the caffe solver and reshape the network
        solver = self.init_caffe(self.solver_file, dict_shapes)
        
#         Start the fetch thread - reading from the in_queue and stores to buff
#         thr_fetcher = threading.Thread(target=self.thr_fetch,
#                                        args=(self.buffer, net_data_shape,
#                                              net_label_shape))
#         thr_fetcher.daemon = True
#         thr_fetcher.start()
#      
#         for batch in iter(self.buffer.get, None):
#             start_iter = time.clock()
#             net_blobs = solver.net.blobs
#             net_blobs['data'].data[...] = batch['data_batch']
#             net_blobs['label'].data[...] = batch['label_batch']
#                   
#             start_train = time.clock()
#             solver.step(1)
#             stop_train = time.clock()
#             self.log.info("Train time: {}".format(stop_train-start_train))
#             self.log.info("Iteration time: {}".format(stop_train-start_iter))
#             self.log.info("Iteration: {}".format(solver.iter))
#             if solver.iter == self.max_iter:
#                 break
#  
        i_batch = 0
        start = time.clock()  
        for packet in iter(self.in_queue.get, None):
            if i_batch < self.batch_size:
                for packet_item in packet:
                    key, packet_data = packet_item.items()[0]
                    packet_data = packet_data.transpose([2, 0, 1])
                    solver.net.blobs[key].data[i_batch][...] = packet_data
                i_batch += 1
                continue
            stop_fetch = time.clock()
               
            start_train = time.clock()
            solver.step(1)
            stop_train = time.clock()
            
            if solver.iter % 50 == 0:
                self.stat.add_history(solver.net)
                self.stat.print_stats()
       
            self.log.debug("Train time: {}".format(stop_train-start_train))
            self.log.debug("Fetch data time: {}".format(stop_fetch-start))
            self.log.debug("Iteration time: {}".format(stop_train-start))
            self.log.debug("Iteration: {}".format(solver.iter))
            if solver.iter == self.max_iter:
                break
      
            start = time.clock()
                  
            i_batch = 0
            
    def save_filters(self, filters_blob, iteration):
        # First blob in BlobVec filters_blob are weights while the second biases
        filters = filters_blob.copy()
        filters -= filters.min()
        filters /= filters.max()
        n_filters = filters.shape[0]
        filter_size = filters.shape[2]+1 # add the border
        
        f_tmp = np.sqrt(n_filters)
        filters_x = np.int(f_tmp)
        filters_y = np.int(np.ceil(f_tmp))
        if (filters_x * filters_y) < n_filters:
            filters_x = (filters_x+1) 
        else:
            filters_x
        
        filter_array_size = np.asarray((filters_x, filters_y), np.int)
        filter_img = np.ones(filter_array_size*filter_size, np.float32)
        
        for i_filter in xrange(0, n_filters):
            cnn_filter = np.pad(np.squeeze(filters[i_filter]),
                                pad_width=((1,0),(1,0)), mode='constant',
                                constant_values=1) 
            unravel_index = np.unravel_index(i_filter, filter_array_size)
            x_y_filter_image = np.asarray(unravel_index)*filter_size
            index_x = x_y_filter_image[0]
            index_y = x_y_filter_image[1]
            filter_img[index_x:index_x+filter_size,
                       index_y:index_y+filter_size ] = cnn_filter
          
        cv2.imwrite("filters_{:06d}.png".format(iteration),
                    (filter_img*255).astype(np.uint))
        #cv2.resize(filter_img*255, (0,0), fx=8, fy=8,
        #            interpolation=cv2.INTER_NEAREST).astype(np.uint) )

class ActivationStat(object):
    
    def __init__(self, net, historySize=20):
        self.historySize = historySize
        dict_blobs = [(blob, []) for blob in net.blobs]
        self.history = collections.OrderedDict(dict_blobs)
        self.log = logging.getLogger(__name__ + ".Stats")

    def add_history(self, net):
        for blob in net.blobs:
            s = net.blobs[blob].data.shape
            if len(s) == 4:
                dims = (0,2,3)
            elif len(s) == 2:
                dims = (0)
            else:
                continue
            self.history[blob].append(np.average(net.blobs[blob].data > 0, dims))
            if len(self.history[blob]) > self.historySize:
                self.history[blob] = self.history[blob][1:]
                  
        for l in net.params:
            if len(net.params[l][0].data.shape) == 4:
                energy = np.sum( net.params[l][0].data**2, (1,2,3))**0.5
                self.log.debug(' ENERGY {} {} {}'.format(l, np.average(energy),
                                                np.std(energy)))

        for l in net.blobs:
            mean = np.average(net.blobs[l].data)
            sdev = np.std(net.blobs[l].data)
            self.log.debug(' BLOB {} {} {}'.format(l, mean, sdev))

    def print_stats(self):
        for blob in self.history:
            try:
                avrg = np.average( np.asarray(self.history[blob]), 0)
                bins=[-10000, 0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8,
                      0.9, 0.99, 10000]
                hist, bins = np.histogram(avrg, bins)
                hist = hist * (1.0/np.sum(hist))
                histString = ' '.join(['%4d' % int(x*100+0.5) for x in hist])
                self.log.info(' {}\t{}'.format(blob, histString))
            except:
                pass
