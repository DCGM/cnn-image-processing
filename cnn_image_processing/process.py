from __future__ import print_function

import multiprocessing
import logging

from utilities import ContinuePipeline, TerminatePipeline


class Process(multiprocessing.Process):
    """
    Encapsulates computing pipeline composed of list of lists of filters.

    Note: While this creates a new process there is a problem to read
    data from the standard input in case of using fileinput.input().
    """

    def __init__(self, pipeline, name):
        """
        Provider constructor
        Args:
          pipeline: pipeline of filters which is continuously called
        """
        super(Provider, self).__init__()
        self.log = logging.getLogger(__name__ + "." + type(self).__name__)
        self.daemon = True  # Kill yourself if parent dies
        self.pipeline = pipeline
        self.name = name

    def run_pipeline(self):
        """
        Run all filters in pipeline.
        """
        packets = [{}] * len(self.pipeline[0])
        for layer in self.pipeline:
            previous = {}
            newPackets = []
            for packet, pfilter in zip(packets, layer):
                newPackets.extend(pfilter(packet, previous))
            packets = newPackets

    def run(self):
        """
        Start the pipeline lop.
        """
        self.log.info(' started.')
        while True:
            try:
                self.run_pipeline()
            except ContinuePipeline:
                pass
            except TerminatePipeline:
                self.log.info('Terminating pipeline.')
                break

