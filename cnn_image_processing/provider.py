'''
Created on May 27, 2016

@author: isvoboda
'''

from __future__ import print_function

import logging
import multiprocessing
from .filters import Packet, FilterTArg


class Process(multiprocessing.Process):

    """
    Exec all the Filters
    """

    def __init__(self, file_list=None, out_queue=None, tfilters=None):
        """
        Provider constructor
        Args:
            queue_size: The between process queue max n_filters.
            reader: Reader parsing the elements per line.
            loop: Boolean True read the file in loop, False read the file once.
        """
        super(Process, self).__init__()
        self.daemon = True  # Kill yourself if parent dies
        self.file_list = file_list
        self.out_queue = out_queue
        self.tfilters = tfilters
        self.tfilters[0][0].init_data(data=self.file_list)
        self.log = logging.getLogger(".".join([__name__, type(self).__name__]))

    def run_tfilters(self):
        """
        Run the tuple filters
        """
        packets = [None] * len(self.tfilters[0])
        try:
            for tfilter in self.tfilters:
                arg = Packet()
                for i_ftr, ftr in enumerate(tfilter):
                    ret_val = ftr(FilterTArg(packets[i_ftr], arg))
                    if type(ret_val) is list:
                        packets = [val.packet for val in ret_val]
                    else:
                        packet, arg = ret_val
                        packets[i_ftr] = packet
            return True
        except StopIteration:
            self.log.info("End")
            return False
        except IOError:
            return False
        else:
            return False

    def run(self):
        """
        Run in separated process
        """
        self.log.info("Begin")
        run = True
        while (run):
            run = self.run_tfilters()


class Provider(multiprocessing.Process):

    """
    Exec all the TupleReaders

    Note: While this creates a new process there is a problem to read
    data from the standard input in case of using fileinput.input().
    """

    def __init__(self, file_list=None, out_queue=None, t_readers=None,
                 loop=False):
        """
        Provider constructor
        Args:
          queue_size: The between process queue max n_filters.
          reader: Reader parsing the elements per line.
          loop: Boolean True read the file in loop, False read the file once.
        """
        super(Provider, self).__init__()
        self.daemon = True  # Kill yourself if parent dies
        self.file_list = file_list
        self.out_queue = out_queue
        self.t_readers = t_readers
        self.loop = loop
        self.log = logging.getLogger(__name__ + ".Provider")

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
            self.log.debug("Number of tuple readers: {}".
                           format(len(self.t_readers)))
            self.log.debug("Number of readers in tuple reader: {}".
                           format([reader.n_filters() for reader in
                                   self.t_readers]))
        else:
            self.log.warning("No tuple reader defined.")

        self.log.debug("File list: {}".format(self.file_list))

        with open(file_list) as flist:
            for line in flist:
                packets = [{'path': path.strip()} for path in line.split()]
                try:
                    packets = self.run_treaders(packets, t_readers)
                except ValueError as val_e:
                    self.log.error(val_e)
                    continue
                self.log.debug("Paths: {}".format(line[0:-1]))
                if self.out_queue is not None:
                    self.out_queue.put(packets)

    def run(self):
        """
        Call the provide_loop once or in loop acording the loop flag.
        run is called by the start method in the separate process.
        """
        self.log.info(" started.")
        try:
            if self.loop is not True:
                self.provide_loop(self.t_readers, self.file_list)
                self.log.info("End of file list: {}".format(self.file_list))
                self.log.info(" end.")
            else:
                while True:
                    self.provide_loop(self.t_readers, self.file_list)
        except EnvironmentError as ex:
            self.log.error(ex)

        if self.out_queue is not None:
            self.out_queue.put(None)
