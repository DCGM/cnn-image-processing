from __future__ import print_function

import numpy as np
import unittest

import time
import os

from ..process import Process
from ..creator import Creator
from ..filters import zmqQueues
from ..utilities import Configurable
import timeout_decorator

THIS_DIR = os.path.dirname(os.path.abspath(__file__))

class test_Creator(unittest.TestCase):
    @timeout_decorator.timeout(1)
    def test_simple_pipeline(self):
        pipeline = [
            [{'ListFileReader':
                {'file_name': os.path.join(THIS_DIR, 'images/data.list')}}],
            [{'TupleReader': {}}, {'TupleReader': {}}],
            [{'HorizontalPassPackets': {}}, {'HorizontalPassPackets': {}}]
        ]
        pip = Creator.parse_pipeline(pipeline)
        self.assertIsInstance(pip, list)
        self.assertEqual(len(pip), len(pipeline))
        for filters, config in zip(pip, pipeline):
            self.assertEqual(len(filters), len(config))
            for f in filters:
                self.assertIsInstance(f, Configurable)


class test_Process(unittest.TestCase):
    @timeout_decorator.timeout(1)
    def test_simple_pipeline(self):
        inConfig = {'url': 'ipc://pipeline_tst', 'bind': False, 'blocking': False}
        inQueue = zmqQueues.InQueueZMQ(inConfig)

        pipeline = [
            [{'ListFileReader':
                {'file_name': os.path.join(THIS_DIR, 'images/data.list'), 'loop': False}}],
            [{'TupleReader': {}}, {'TupleReader': {}}],
            [{'HorizontalPassPackets': {}},
                {'OutQueueZMQ': {'url': 'ipc://pipeline_tst', 'bind': True}}]
        ]
        pipeline = Creator.parse_pipeline(pipeline)
        process = Process(pipeline, name='test')
        self.assertEqual(process.name, 'test')
        process.run()
        time.sleep(0.01)

        for i in range(3):
            packetsOut = inQueue(None, {})

        packetsOut = inQueue(None, {})
        self.assertEqual(packetsOut, [{},{}])



'''class Process(multiprocessing.Process):
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
'''
