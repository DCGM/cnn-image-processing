'''
Created on Apr 11, 2016

@author: isvoboda, ihradis

'''

from __future__ import print_function
import sys
import os
# import threading
import Queue
from Queue import Full, Empty
import multiprocessing
import logging
import filters.data_filters as df
import numpy as np
import cv2
import time
import collections
import caffe

MODULE_LOGGER = logging.getLogger(__name__)

class Creator(object):
    """
    Create the readers and filters according the configuration file.
    """
    def __init__(self, config):
        self.config = config
        self.f_create = ObjectFactory.create_object
    
    def parse_filters(self, l_filters):
        """
        Parse list of filters.
        """
        new_filters = []
        for fil in l_filters:
            (fil_id, fil_params), = fil.items()
            if fil_params != None:
                new_filters.append(self.f_create(fil_id, **fil_params))
            else:
                new_filters.append(self.f_create(fil_id))
        
        return new_filters 
    
    def parse_tuples(self, tuples):
        """
        Parse list of filter tuples.
        """
        tup_filters = []
        filters = None
        for tup_f in tuples:
            parameters = dict()
            for (key_id, val) in tup_f.items():
                for (param_id, param_val) in val.items():
                    if param_id == "Filters" or param_id == "Readers":
                        filters = self.parse_filters(param_val)
                    elif param_id == "Parameters" and param_val is not None:
                        parameters.update(param_val)
                # Fixme: if parameters == None, than fails
                parameters['filters'] = filters
                assert parameters != None
                t_fils = self.f_create(key_id, **parameters)
                tup_filters.append(t_fils)

        return tup_filters

    def create_provider(self):
        """
        Creates provider.
        """
        tuple_filters = self.parse_tuples(self.config['Provider']['TFilters'])
        parameters = self.config['Provider']['Parameters']
        return DataProvider(t_readers=tuple_filters, **parameters)
        
    def create_processing(self):
        """
        Creates processing
        """
        tuple_filters = self.parse_tuples(self.config['Processing']
                                          ['TFilters'])
        parameters = self.config['Processing']['Parameters']
        return DataProcessing(t_filters=tuple_filters, **parameters)
    
    def create_training(self):
        """
        Creates the trainer.
        """
        return Trainer(**self.config['Trainer'])
    
class RADequeue(object):

    """
    Round buffer with random access memory.
    """

    def __init__(self, max_size=5):
        self.max_size = max_size
        self.i_index = 0
        self.size = 0
        self.buffer = [None] * self.max_size
        self.it_id = 0

    def append(self, obj):
        """
        Append the obj to the RADequeue.
        """
        if self.size == self.max_size:
            raise Full  # "RADequeue full, invalid attempt to assign a value."

        self.buffer[self.i_index % self.max_size] = obj
        self.i_index += 1
        self.size += 1

    def append_round(self, obj):
        """
        Append the item obj and if necessary pop the first in item.
        """
        ret = None
        if self.size == self.max_size:
            ret = self.pop()
        self.append(obj)
        return ret

    def pop(self):
        """
        Pop the first in obj.
        """
        if self.size == 0:
            raise Empty  # "RADequeue empty, invalid attempt to pop a value."
        ret = self[0]
        self.size -= 1
        return ret

    def __getitem__(self, key):
        if 0 > key or self.size < key:
            raise IndexError  #"RADequeue attempt to access key out of bounds."
        i = (self.i_index - self.size + key) % self.max_size
        return self.buffer[i]

    def __setitem__(self, key, value):
        if 0 > key or self.size < key:
            raise IndexError  #"RADequeue attempt to set key out of bounds."
        i = (self.i_index - self.size + key) % self.max_size
        self.buffer[i] = value

    def __iter__(self):
        self.it_id = 0
        return self

    def next(self):
        """
        Return next obj.
        """
        if self.it_id == self.size:
            raise StopIteration
        key = self.it_id
        self.it_id += 1
        return self.__getitem__(key)

class ImageReader(object):
    """
    Reads several types of images via OpenCV.
    Always returns the float 3dim numpy array.
    """
    def __init__(self):
        self.log = logging.getLogger(__name__ + ".ImageReader")

    def read(self, path):
        """
        Loads and returns the image on path.
        """
        img = None
        try:
            img = cv2.imread(path, -1).astype(np.float)
            if img is None:
                self.log.error("Could not read image: {}".format(path))
                return None
        except cv2.error as err:
            self.log.error("cv2.error: {}".format(str(err)))
        except:
            self.log.error("UNKNOWN: {}".format(sys.exc_info()[0]))
        if len(img.shape) == 2:
            img = img.reshape(img.shape[0],img.shape[1],1)
        return img
    
    def __call__(self, path):
        """
        Returns the image
        """
        return self.read(path)

class ImageWriter(object):
    """
    Write images.
    """

    def __init__(self, d_path=None):
        """
        ImageWriter constructor
        Args:
          d_path: directory path where write the image.
        """
        self.d_path = d_path if d_path != None else os.getcwd()
        self.log = logging.getLogger(__name__ + ".ImageWriter")

    def __call__(self, file_name, img):

        img_name = os.path.join(self.d_path, file_name)
        cv2.imwrite(img_name, img)

class ImageX8Reader(ImageReader):
    """
    Reads several types of images via OpenCV.
    Crop the images to have the size divideable by 8.
    Pivot for cropping is the image left up corner.
    """
    log = logging.getLogger(__name__ + ".ImageX8Reader")

    @classmethod
    def read(cls, path):
        """
        Loads and returns the cropped image size divideable by 8.
        """
        img = super(ImageX8Reader, cls).read(path)
        if img != None:
            return ImageX8Reader.crop(img)
        else:
            return img

    @classmethod
    def crop(cls, img):
        """
        Crop the img to be 8x divisible.
        """
        crop_c = [i - (i % 8) for i in img.shape[0:2]]
        return img[0:crop_c[0], 0:crop_c[1], ...]

class CoefNpyTxtReader(object):
    """
    Reads the jpeg-coefs in numpy array txt file format.
    Always returns float Xdim numpy array.
    """
    def __init__(self, n_channels=64):
        self.n_channels = n_channels
        self.log = logging.getLogger(__name__ + ".CoefNpyTxtReader")

    def read(self, path):
        """
        Reads the numpy txt array file and returns the numpy array.
        """
        data_array = None
        try:
            data_array = np.loadtxt(fname=path, dtype=np.float)
            if data_array is None:
                self.log.error("Could not read data: {}".format(path))
            data_array = data_array.reshape([self.n_channels, -1,
                                             data_array.shape[1]])
            data_array = data_array.transpose([1, 2, 0])  # [y,x,z]
        except IOError as ioe:
            self.log.error("IOError: {}".format(str(ioe)))

        return data_array
    
    def __call__(self, path):
        """
        Return the JPEG coefs stored in the txt file in path.
        """
        return self.read(path)

class CoefNpyTxtWriter(object):
    """
    Write numpy nd arrays of jpeg coefs.
    """
    def __init__(self, d_path=None):
        """
        ImageWriter constructor
        Args:
          d_path: directory path where write the coeficients.
        """
        self.d_path = d_path if d_path != None else os.getcwd()
        self.log = logging.getLogger(__name__ + "CoefNpyTxtWriter")

    def __call__(self, file_name, coef):
        coef_file_name = os.path.join(self.d_path, file_name)
        coef_w = coef.transpose([2, 0, 1]).reshape([-1, coef.shape[1]])
        np.savetxt(fname=coef_file_name, X=coef_w, fmt="%3d")

# Deprecated not in use, see the TFilter instead
class TReader(object):
    """
    Read tuple according the file_list.
    Tuple size reflects the number of elements per line separated by whitespace.
    """

    def __init__(self, readers=None):
        """
        TupleReader constructor
        Args:
          readers: Tuple of readers in the same order as the elements
          they read.
        """
        self.readers = readers
        self.log = logging.getLogger(__name__ + ".TupleReader")

    def size(self):
        if self.readers is not None:
            return len(self.readers)
        else:
            return 0 

    def read(self, line):
        """
        Parse and read the input line.
        Args:
          line: string with the elements separated with whitespace.
        """
        start = time.clock()
        s_line = line.split()
        assert len(self.readers) == len(s_line)
        zip_reader_data = zip(self.readers, s_line)
        t_data = tuple(val for val in
                      ((zip_rd[0].read(zip_rd[1]))
                       for zip_rd in zip_reader_data) if val != None)

        t_diff = time.clock() - start

        self.log.debug("Loading Time: {}".format(t_diff))
        return t_data if len(t_data) == len(self.readers) else None

    def __call__(self, line):
        return self.read(line)

class DataProvider(multiprocessing.Process):
    """
    Exec all the TupleReaders

    Note: While this creates a new process there is a problem to read
    data from the standard input in case of using fileinput.input().
    """

    def __init__(self, file_list=None, out_queue=None , t_readers=None,
                loop=False):
        """
        DataProvider constructor
        Args:
          queue_size: The between process queue max size.
          reader: Reader parsing the elements per line.
          loop: Boolean True read the file in loop, False read the file once.
        """
        super(DataProvider, self).__init__()
        self.daemon = True  # Kill yourself if parent dies
        self.file_list = file_list
        self.out_queue = out_queue
        self.t_readers = t_readers
        self.loop = loop
        self.log = logging.getLogger(__name__ + ".DataProvider")

    def run_treaders(self, t_data, t_readers):
        """
        Run the tuple readers
        """
        for t_reader in t_readers:  # exec readers
            t_data = t_reader(t_data)
            if t_data == None:
                raise ValueError("Returned None by: {}".format(t_reader))
        return t_data

    def provide_loop(self, t_readers, file_list):
        """
        Loop over file_list and call the run_treaders.
        Args:
          t_readers: tuple of readers/filters sequentially applied.
          file_list: list of files to process.
        """
        if t_readers is not None:
            self.log.info("Number of tuple readers: {}".
                          format(len(self.t_readers)))
            self.log.info("Number of readers in tuple reader: {}".
                          format([ reader.size() for reader in self.t_readers]))
        else:
            self.log.warning("No tuple reader defined.")
        
        self.log.info("File list: {}".format(self.file_list))

        with open(file_list) as flist:
            for line in flist:
                t_data = line
                try:
                    t_data = self.run_treaders(line.split(), t_readers)
                except ValueError as val_e:
                    self.log.error(val_e)
                    continue
                self.log.debug("Paths: {}".format(line[0:-1]))
                self.out_queue.put(t_data)

    def run(self):
        """
        Call the provide_loop once or in loop acording the loop flag.
        run is called by the start method in the separate process.
        """
#     assert(self.t_readers != None)
        self.log.info("started.")

        if self.loop is not True:
            self.provide_loop(self.t_readers, self.file_list)
        else:
            while True:
                self.provide_loop(self.t_readers, self.file_list)

        self.out_queue.put(None)
        self.log.info("end.")

class DataProcessing(multiprocessing.Process):
    """
      DataProcessing reads the data from in_queue, stores them in to the
      shift buffer and performs all the t_filters on the data in the buffer.
      The buffer is shifted after t_filters processed.
    """

    def __init__(self, in_queue=None, out_queue=None, buffer_size=100,
                t_filters=None, samples=1, RNG=None):
        """
        DataProcessing constructor
        Args:
          in_queue: The queue the data are read from.
          queue_size: Size of the output queue the processed data are
                     pushed to.
          buffer_size: Size of the buffer of read data to process.
          t_filters: Tuple of filters applied on the read data.
                     Filters are concatenated according the order in the tuple.
          samples: Count of filter application before shifting the data
                     in buffer.
          RNG: Random state Generator
        """
        super(DataProcessing, self).__init__()
        self.daemon = True  # Kill yourself if parent dies
        self.in_queue = in_queue
        self.out_queue = out_queue
        self.buffer_size = buffer_size
        self.t_filters = t_filters
        self.samples = samples
        self.buffer = RADequeue(max_size=buffer_size)
        self.rng = RNG if RNG != None else np.random.RandomState(5)
        self.log = logging.getLogger(__name__ + ".DataProcessing")

    def load_buffer(self):
        """
        Loads the buffer.
        """
        self.log.info("Initializing - loading buffer.")
        for data in iter(self.in_queue.get, None):
            self.buffer.append_round(data)
            if self.buffer.size == self.buffer_size:
                break

    def run_xtimes_tfilters(self):
        """
        Run samples times the tuple filters with random data from
        the buffer. The results is pushed into the output data queue.
        """
        for _ in xrange(self.samples):
            i_buffer = self.rng.randint(0, self.buffer.size)
            rn_data = self.buffer[i_buffer]
            for t_filter in self.t_filters:
                rn_data = t_filter(rn_data)
            
            if rn_data != None:
                self.out_queue.put(rn_data)

    def run(self):
        self.log.info("started.")
        # Load the buffer
        self.load_buffer()
        # Fetch from the queue
        for data in iter(self.in_queue.get, None):
            self.log.debug("Received data.")
            start = time.clock()
            self.run_xtimes_tfilters()
            t_dif = time.clock() - start
            self.log.debug("Processing Time: {}".format(t_dif))
            self.buffer.append_round(data)
        # Flush out the buffer
        while self.buffer.size != 0:
            self.run_treaders()
            self.buffer.pop()

        self.log.info("end.")
        self.out_queue.put(None)

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

def decode_dct(coefs):
    """
    Compute the inverse dct of the coefs.
    Coefs is the x * y * 64 NDarray where 1*1*64 is one block of coefficients
    equivalent to the 8*8 image pixels.
    """
#     coefs_shape = coefs.transpose()
    img = np.zeros([ dim * 8 for dim in coefs.shape[0:2]], dtype=np.double)
    step = 8
    for y_coef in xrange(coefs.shape[0]):
        for x_coef in xrange(coefs.shape[1]):
            in_x = step*x_coef
            in_y = step*y_coef 
            img[in_y:in_y+step, in_x:in_x+step] = \
                cv2.idct(coefs[y_coef, x_coef].reshape([8,8])) * 1024

    img += 128
    return img

class ObjectFactory(object):

    """
    Factory class to create several object.
    """
    factories = { 'Pass': df.Pass,
                  'TFilter':  df.TFilter,
                  'TCropCoef8ImgFilter': df.TCropCoef8ImgFilter,
                  'TReader': df.TFilter,
                  'CoefNpyTxtReader': CoefNpyTxtReader,
                  'ImageReader': ImageReader,
                  'Crop': df.Crop,
                  'LTCrop': df.LTCrop,
                  'Label': df.Label,
                  'Mul': df.Mul,
                  'Sub': df.Sub }  # Static attribute

    @staticmethod
    def create_object(id_object, **kwargs):
        """
        Creates the object according the id_filter.
        In case the id_object has not yet been registered than add it.
        """
        if not ObjectFactory.factories.has_key(id_object):
            # ToDo eval is not safe - anything could be inserted in.
            ObjectFactory.factories[id_object] = \
                                           ObjectFactory.get_class(id_object)
            print(id_object)
        return ObjectFactory.factories[id_object](**kwargs)

    @staticmethod
    def get_class(class_name):
        """
        Returns the class of the name class_name

        Args:
          class_name: name of the class to be referenced (eg module.class)
        Return:
          The reference to the class of name class_name
        """
        parts = class_name.split(".")
        module = ".".join(parts[:-1])
        print(module)
        mod = __import__(module)
        for component in parts[1:]:
            mod = getattr(mod, component)
        return mod
