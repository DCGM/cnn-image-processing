from __future__ import print_function

import copy
import numpy as np
import unittest
import os
from ..filters import readers
from ..utilities import TerminatePipeline, ContinuePipeline
from .. import FilterFactory
from .. import utilities

THIS_DIR = os.path.dirname(os.path.abspath(__file__))


class test_FilterFactory(unittest.TestCase):
    def test_registered_filters(self):
        reader_names = [ 'CoefNpyTxtReader', 'ImageX8Reader',
            'TupleReader', 'ImageReader', 'ListFileReader',
            'HorizontalPassPackets', 'Preview',
            'Label', 'MulAdd', 'Pass', 'JPEG', 'Resize', 'CustomFunction',
            'CentralCrop', 'Round']
        for name in reader_names:
            self.assertIn(name, FilterFactory.factories.keys())

    def test_readers(self):
        configs = [
            {'HorizontalPassPackets': {}},
            {'TupleReader': {}},
            {'ImageReader': {}},
            {'ListFileReader': {'file_name': os.path.join(THIS_DIR, 'images/list.txt')}}]
        for config in configs:
            id_object = config.keys()[0]
            reader = FilterFactory.create_object(id_object=id_object, config=config[id_object])
            self.assertIsInstance(reader, utilities.Configurable)


class test_ListFileReader(unittest.TestCase):
    def test_read(self):
        config = {'file_name': os.path.join(THIS_DIR, 'images/list.txt')}
        reader = readers.ListFileReader(config)
        packets = reader({}, {})
        self.assertEqual(len(packets), 2)
        self.assertEqual(packets[0]['data'], '110200009.JPG')
        self.assertEqual(packets[1]['data'], '110200010.JPG')
        packets = reader({}, {})
        self.assertEqual(len(packets), 2)
        self.assertEqual(packets[0]['data'], '110200011.JPG')
        self.assertEqual(packets[1]['data'], '110200012.JPG')

    def test_no_loop(self):
        config = {'file_name': os.path.join(THIS_DIR, 'images/list.txt'), 'loop': False}
        reader = readers.ListFileReader(config)
        reader({}, {})
        reader({}, {})
        self.assertRaises(TerminatePipeline, reader, {}, {})

    def test_loop(self):
        config = {'file_name': os.path.join(THIS_DIR, 'images/list.txt'), 'loop': True}
        reader = readers.ListFileReader(config)
        reader({}, {})
        reader({}, {})
        self.assertRaises(ContinuePipeline, reader, {}, {})
        reader({}, {})

    def test_loop_optional(self):
        config = {'file_name': os.path.join(THIS_DIR, 'images/list.txt')}
        reader = readers.ListFileReader(config)
        reader({}, {})
        reader({}, {})
        self.assertRaises(ContinuePipeline, reader, {}, {})
        reader({}, {})

    def test_empty_file(self):
        config = {'file_name': os.path.join(THIS_DIR, 'images/empty.list')}
        self.assertRaises(IOError, readers.ListFileReader, config)

    def test_wrong_file_path(self):
        config = {'file_name': os.path.join(THIS_DIR, 'images/list.txtx')}
        self.assertRaises(IOError, readers.ListFileReader, config)

class test_ImageReader(unittest.TestCase):

    def test_read_color(self):
        config = {'grayscale': False}
        reader = readers.ImageReader(config)
        path = os.path.join(THIS_DIR, 'images/11020009.JPG')
        packet = {'data': path, 'some other data': True}
        previous = {'something': True}
        previousBackup = copy.deepcopy(previous)
        packet = reader(packet, previous)
        self.assertIsInstance(packet, list)
        packet = packet[0]
        self.assertIn('some other data', packet)
        self.assertEqual(packet['some other data'], True)
        self.assertIn('data', packet)
        self.assertEqual(packet['data'].shape, (600, 800, 3))
        self.assertIn('path', packet)
        self.assertEqual(packet['path'], path)
        self.assertEqual(previousBackup, previous)

    def test_read_grayscale(self):
        config = {'grayscale': True}
        reader = readers.ImageReader(config)
        path = os.path.join(THIS_DIR, 'images/11020010.JPG')
        packet = {'data': path, 'some other data': True}
        previous = {'something': True}
        previousBackup = copy.deepcopy(previous)
        packet = reader(packet, previous)
        self.assertIsInstance(packet, list)
        packet = packet[0]
        self.assertIn('some other data', packet.keys())
        self.assertEqual(packet['some other data'], True)
        self.assertIn('data', packet)
        self.assertEqual(packet['data'].shape, (600, 800, 1))
        self.assertEqual(packet['data'].dtype, np.float32)
        self.assertIn('path', packet)
        self.assertEqual(packet['path'], path)
        self.assertEqual(previousBackup, previous)

    def test_read_color_default(self):
        config = {}
        reader = readers.ImageReader(config)
        path = os.path.join(THIS_DIR, 'images/11020010.JPG')
        packet = {'data': path, 'some other data': True}
        previous = {'something': True}
        previousBackup = copy.deepcopy(previous)
        packet = reader(packet, previous)
        self.assertIsInstance(packet, list)
        packet = packet[0]
        self.assertIn('some other data', packet.keys())
        self.assertEqual(packet['some other data'], True)
        self.assertIn('data', packet)
        self.assertEqual(packet['data'].shape, (600, 800, 3))
        self.assertEqual(packet['data'].dtype, np.float32)
        self.assertIn('path', packet)
        self.assertEqual(packet['path'], path)
        self.assertEqual(previousBackup, previous)

    def test_no_file(self):
        config = {}
        reader = readers.ImageReader(config)
        path = os.path.join(THIS_DIR, 'does_not_exist.jpg')
        self.assertRaises(ContinuePipeline, reader, packet={'data': path}, previous={})

    def test_wrong_file(self):
        config = {}
        reader = readers.ImageReader(config)
        path = os.path.join(THIS_DIR, 'images/list.txt')
        self.assertRaises(ContinuePipeline, reader, packet={'data': path}, previous={})


class test_TupleReader(unittest.TestCase):
    def test_read(self):
        reader = readers.TupleReader({})
        data = np.asarray([0.25, -32, +1.6e10, 5.9604644775390625E-8])
        dataString = ','.join([str(x) for x in data])
        packet = {'data': dataString, 'some other data': True}
        previous = {'something': True}
        previousBackup = copy.deepcopy(previous)
        packets = reader(packet, previous)
        self.assertIsInstance(packets, list)
        packet = packets[0]
        self.assertIn('some other data', packet.keys())
        self.assertEqual(packet['some other data'], True)
        self.assertIn('data', packet.keys())
        self.assertEqual(packet['data'].shape, (1, 1, 4))
        self.assertEqual(packet['data'].dtype, np.float32)
        self.assertEqual(packet['data'].reshape(-1).tolist(), data.tolist())
        self.assertEqual(previousBackup, previous)

    def test_wrong_string(self):
        reader = readers.TupleReader({})
        data = '1,3,5,8,1.asdf'
        self.assertRaises(ContinuePipeline, reader, packet={'data': data}, previous={})


if __name__ == '__main__':
    unittest.main()
