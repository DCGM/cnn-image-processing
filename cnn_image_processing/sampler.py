'''
Created on May 27, 2016

@author: isvoboda
'''

from __future__ import print_function

import multiprocessing
import logging
import time
import numpy as np
from .utils import RoundBuffer

class Sampler(multiprocessing.Process):
    """
      Sampler reads the data from in_queue, stores them in to the
      shift buffer and performs all the t_filters on the data in the buffer.
      The buffer is shifted after t_filters processed.
    """

    def __init__(self, in_queue=None, out_queue=None, buffer_size=100,
                t_filters=None, samples=1, RNG=None):
        """
        Sampler constructor
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
        super(Sampler, self).__init__()
        self.daemon = True  # Kill yourself if parent dies
        self.in_queue = in_queue
        self.out_queue = out_queue
        self.buffer_size = buffer_size
        self.t_filters = t_filters
        self.samples = samples
        self.buffer = RoundBuffer(max_size=buffer_size)
        self.rng = RNG if RNG != None else np.random.RandomState(5)
        self.log = logging.getLogger(__name__ + ".Sampler")

    def load_buffer(self):
        """
        Loads the buffer.
        """
        self.log.info("Initializing - loading buffer.")
        for packets in iter(self.in_queue.get, None):
            self.buffer.append_round(packets)
            if self.buffer.size == self.buffer_size:
                break

    def run_xtimes_tfilters(self):
        """
        Run samples times the tuple filters with random data from
        the buffer. The results is pushed into the output data queue.
        """
        for _ in xrange(self.samples):
            i_buffer = self.rng.randint(0, self.buffer.size)
            rn_packets = self.buffer[i_buffer]
            for t_filter in self.t_filters:
                rn_packets = t_filter(rn_packets)
            
            if rn_packets != None:
                self.out_queue.put(rn_packets)

    def run(self):
        self.log.info("started.")
        # Load the buffer
        self.load_buffer()
        # Fetch from the queue
        for packets in iter(self.in_queue.get, None):
            self.log.debug("Received data.")
            start = time.clock()
            self.run_xtimes_tfilters()
            t_dif = time.clock() - start
            self.log.debug("Processing Time: {}".format(t_dif))
            self.buffer.append_round(packets)
        # Flush out the buffer
        while self.buffer.size != 0:
            self.run_xtimes_tfilters()
            self.buffer.pop()

        self.log.info("end.")
        self.out_queue.put(None)
