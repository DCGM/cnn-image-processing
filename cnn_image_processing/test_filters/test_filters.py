from __future__ import print_function

import copy
import numpy as np
import unittest
import os
import cv2
from ..filters import filters
from ..utilities import TerminatePipeline, ContinuePipeline
from .. import FilterFactory
from .. import utilities

THIS_DIR = os.path.dirname(os.path.abspath(__file__))


class test_Buffer(unittest.TestCase):
    def test_run(self):
        packets = [{'data': 1}, {'data': 2}, {'data': 3}]
        previous = {'packets': packets[0:2]}
        packetIn = packets[2]
        config = {'size': 1}
        filter_object = filters.Buffer(config)
        packetsOut = filter_object(packetIn, previous)
        self.assertIsInstance(packetsOut, list)
        self.assertEqual(packets, packetsOut)

        packetsOut = filter_object(None, {})
        self.assertIsInstance(packetsOut, list)
        self.assertEqual(packets, packetsOut)

    def test_randomness(self):
        count = 4
        iterations = 200
        packets = [{'data': x} for x in range(count)]
        config = {'size': count}
        filter_object = filters.Buffer(config)
        for packet in packets:
            packetsOut = filter_object(packet, {})

        good = {}
        for i in range(iterations):
            packetsOut = filter_object(None, {})
            print(packetsOut[0]['data'])
            good[packetsOut[0]['data']] = 1
        self.assertEqual(len(good), count)

    def test_fifo(self):
        count = 10
        size = 4
        iterations = 200
        packets = [{'data': x} for x in range(count)]
        config = {'size': size}
        filter_object = filters.Buffer(config)
        for packet in packets:
            packetsOut = filter_object(packet, {})

        good = {}
        for i in range(iterations):
            packetsOut = filter_object(None, {})
            good[packetsOut[0]['data']] = 1
        for packet in packets[count-size:]:
            self.assertIn(packet['data'], good)
        for packet in packets[0:count-size]:
            self.assertNotIn(packet['data'], good)

    def test_empty_read(self):
        config = {'size': 5}
        filter_object = filters.Buffer(config)
        self.assertRaises(ContinuePipeline, filter_object, {}, {})


class test_HorizontalPassPackets(unittest.TestCase):
    def test_pass_through(self):
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

    def test_no_pass_through(self):
        data = np.zeros((10, 10))
        packetIn = {'data': data, 'some other data': True}
        config = {'pass_through': False}
        filter_object = filters.HorizontalPassPackets(config)
        previous = {}
        packetsOut = filter_object(packetIn, previous)
        packetsOut = filter_object(packetIn, previous)
        self.assertIsInstance(packetsOut, list)
        self.assertEqual(len(packetsOut), 0)

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


class test_JPEG(unittest.TestCase):
    def test_normal(self):
        data = np.random.ranf((48, 48, 1))
        packetIn = {'data': data}
        quality = 10
        config = {'quality': quality}
        filter_object = filters.JPEG(config)
        previous = {}
        packetsOut = filter_object(packetIn, previous)
        self.assertIsInstance(packetsOut, list)
        packetOut = packetsOut[0]
        self.assertIsNot(packetOut['data'], data)
        self.assertEqual(packetOut['data'].shape, data.shape)
        res, img_str = cv2.imencode('.jpeg', data,
                                    [cv2.IMWRITE_JPEG_QUALITY, quality])
        img = cv2.imdecode(np.asarray(bytearray(img_str), dtype=np.uint8),
                           cv2.IMREAD_UNCHANGED)

        self.assertTrue((img == packetOut['data']).all())


class test_Resize(unittest.TestCase):
    def test_relative_scale(self):
        for data in [np.random.ranf((48, 64, 1)), np.random.ranf((48, 48, 3))]:
            for scale in [0.1, 0.3, 0.5, 1, 2, 3.3]:
                packetIn = {'data': data}
                config = {'scale': scale}
                filter_object = filters.Resize(config)
                previous = {}
                packetsOut = filter_object(packetIn, previous)
                self.assertIsInstance(packetsOut, list)
                packetOut = packetsOut[0]
                self.assertIsNot(packetOut['data'], data)
                newShape = [int(x * scale + 0.5) for x in data.shape]
                newShape[-1] = data.shape[-1]
                newShape = tuple(newShape)
                self.assertEqual(packetOut['data'].shape, newShape)

    def test_absolute_size(self):
        for data in [np.random.ranf((48, 64, 1)), np.random.ranf((48, 48, 3))]:
            for scale in [[10, 10], [10, 68], [256, 128]]:
                packetIn = {'data': data}
                config = {'fixedSize': scale}
                filter_object = filters.Resize(config)
                previous = {}
                packetsOut = filter_object(packetIn, previous)
                self.assertIsInstance(packetsOut, list)
                packetOut = packetsOut[0]
                self.assertIsNot(packetOut['data'], data)
                newShape = tuple([scale[1], scale[0], data.shape[2]])
                self.assertEqual(packetOut['data'].shape, newShape)

if __name__ == '__main__':
    unittest.main()
