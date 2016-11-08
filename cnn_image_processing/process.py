from __future__ import print_function

import multiprocessing
import logging

from .creator import Creator
from .utilities import ContinuePipeline, TerminatePipeline, Configurable, parameter


class Process(Configurable, multiprocessing.Process):
    """
    Encapsulates computing pipeline composed of list of lists of filters.

    Note: While this creates a new process there is a problem to read
    data from the standard input in case of using fileinput.input().
    """
    def addParams(self):
        self.params.append(parameter(
            'name', required=True,
            parser=str,
            help='Name identifying the process.'))
        self.params.append(parameter(
            'pipeline', required=True,
            help='Pipeline.'))

    def __init__(self, config):
        """
        Provider constructor
        Args:
          pipeline: pipeline of filters which is continuously called
        """
        multiprocessing.Process.__init__(self)
        Configurable.__init__(self)
        #super(Process, self).__init__()
        self.daemon = True
        self.log = logging.getLogger(__name__ + "." + type(self).__name__)
        self.addParams()
        self.parseParams(config)
        self.pipeline = Creator.parse_pipeline(self.pipeline)


    def run_pipeline(self):
        """
        Run all filters in pipeline.
        """
        packets = [{}] * len(self.pipeline[0])
        for stage_id, layer in enumerate(self.pipeline):
            previous = {}
            newPackets = []
            continueRaised = False
            for filter_id, (packet, pfilter) in enumerate( zip(packets, layer)):
                try:
                    filterOutput = pfilter(packet, previous)
                    newPackets.extend(filterOutput)
                except ContinuePipeline:
                    continueRaised = True
            if continueRaised:
                raise ContinuePipeline
            #self.log.info('PIPELINE {} {}: {}'.format(self.name, stage_id, len(newPackets)))
            packets = newPackets

    def init(self):
        for stage in self.pipeline:
            for f in stage:
                f.init()

    def run(self):
        """
        Start the pipeline lop.
        """
        self.init()
        self.log.info(' started.')
        while True:
            try:
                self.run_pipeline()
            except ContinuePipeline:
                pass
            except TerminatePipeline:
                self.log.info('Terminating pipeline.')
                break

