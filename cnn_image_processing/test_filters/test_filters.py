from __future__ import print_function

import copy
import numpy as np
import unittest
import os
from ..filters import filters
from ..utilities import TerminatePipeline, ContinuePipeline
from .. import FilterFactory
from .. import utilities

THIS_DIR = os.path.dirname(os.path.abspath(__file__))


class test_HorizontalPassPackets(unittest.TestCase):
    def test_run(self):
        data = np.zeros((10, 10))
        packetIn = {'data': data, 'some other data': True}
        filter_object = filters.HorizontalPassPackets({})
        previous = {}
        packetsOut = filter_object(packetIn, previous)
        packetsOut = filter_object(packetIn, previous)
        self.assertIsInstance(packetsOut, list)
        packetOut = packetsOut[0]
        self.assertEqual(packetIn, packetOut)

        self.assertIn('packets', previous.keys())
        self.assertIsInstance(previous['packets'], list)
        self.assertEqual(len(previous['packets']), 2)
        self.assertEqual(previous['packets'][0], packetIn)
        self.assertEqual(previous['packets'][1], packetIn)


class test_Pass(unittest.TestCase):
    def test_run(self):
        data = np.zeros((10, 10))
        packetIn = {'data': data, 'some other data': True}
        filter_object = filters.Pass({})
        previous = {}
        packetsOut = filter_object(packetIn, previous)
        self.assertIsInstance(packetsOut, list)
        packetOut = packetsOut[0]
        self.assertEqual(packetIn, packetOut)
        self.assertEqual(previous, {})


class test_Label(unittest.TestCase):
    def test_run(self):
        data = np.zeros((10, 10))
        packetIn = {'data': data}
        config = {'label_name': 'label'}
        filter_object = filters.Label(config)
        previous = {}
        packetsOut = filter_object(packetIn, previous)
        self.assertIsInstance(packetsOut, list)
        packetOut = packetsOut[0]
        self.assertEqual(packetOut['data'].shape, packetIn['data'].shape)
        self.assertEqual(packetOut['label'], 'label')


class test_MulAdd(unittest.TestCase):
    def test_normal(self):
        data = np.ones((10, 10))
        packetIn = {'data': data}
        config = {'mul': 0.5, 'add': 10}
        filter_object = filters.MulAdd(config)
        previous = {}
        packetsOut = filter_object(packetIn, previous)
        self.assertIsInstance(packetsOut, list)
        packetOut = packetsOut[0]
        self.assertIsNot(packetOut['data'], data)
        self.assertEqual(packetOut['data'].shape, data.shape)
        self.assertTrue((packetOut['data'] == data * 0.5 + 10).all())
        self.assertIn('op', previous.keys())
        newData = previous['op'](data)
        self.assertTrue((newData == data * 0.5 + 10).all())
        self.assertIsNot(newData, data)



if __name__ == '__main__':
    unittest.main()
