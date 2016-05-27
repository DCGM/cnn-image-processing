'''
Created on May 27, 2016

@author: isvoboda
'''

from __future__ import print_function

import multiprocessing
import logging

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
                          format([ reader.size() for reader in
                                  self.t_readers]))
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
