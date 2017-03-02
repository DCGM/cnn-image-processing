'''
Created on May 27, 2016

@author: isvoboda
'''
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.ioff()

import Queue
from Queue import Empty
import multiprocessing
import logging
import time
import numpy as np
import cv2
from collections import defaultdict
import timeout_decorator
import Queue
import threading

import caffe

from .creator import Creator
from .factory import FilterFactory
from .utils import RoundBuffer
from .utilities import parameter, Configurable


def fill_net_input(net, data):
    for inputName in data:
        packet_data = data[inputName]
        if inputName in net.blobs:
            net.blobs[inputName].data[...] = packet_data


def fetch_batch(inQueue, batchSize):
    data = defaultdict(list)
    for i in range(batchSize):
        packets = inQueue({}, {})
        for packet in packets:
            data[packet['label']].append(packet['data'])
    for key in data:
        data[key] = np.concatenate(data[key])

    #data['id'] = data['id'].reshape(-1,1)
    return data


def set_shapes(net, batch):
    """
    Set the inputlayer shape according the data shape
    """
    matched = []
    unmatched = []
    for key, data in batch.items():
        if key in net.blobs:
            print(' Shapes {} {} {}'.format(key, net.blobs[key].data.shape, data.shape))
            net.blobs[key].reshape(*data.shape)
            matched.append(key)
        else:
            unmatched.append(key)
    net.reshape()
    return matched, unmatched


class Tester(Configurable):
    def addParams(self):
        self.params.append(parameter(
            'name', required=True,
            parser=str,
            help='Name of the test.'))
        self.params.append(parameter(
            'batch_size', required=True,
            parser=lambda x: max(int(x), 1),
            help='Size of data batches.'))
        self.params.append(parameter(
            'iterations', required=True,
            parser=lambda x: max(int(x), 1),
            help='Number of iterations in each test pass.'))
        self.params.append(parameter(
            'data_queue', required=True,
            parser=lambda x: x,
            help='Queue from which data should be read.'))
        self.params.append(parameter(
            'evaluators', required=False,
            parser=list,
            help='List of evaluators.'))
        self.params.append(parameter(
            'display', required=False, default=False,
            parser=bool,
            help='Display images.'))
        self.params.append(parameter(
            'write', required=False, default=False,
            parser=bool,
            help='Write images to disk.'))
        self.params.append(parameter(
            'data_layer', required=False, default='data',
            parser=str,
            help='Network input data layer.'))
        self.params.append(parameter(
            'gt_layer', required=False, default='labels',
            parser=str,
            help='Network input ground truth layer.'))
        self.params.append(parameter(
            'result_layer', required=False, default='out',
            parser=str,
            help='Network output layer.'))

    def __init__(self, config):
        super(Tester, self).__init__()
        self.log = logging.getLogger(__name__ + "." + type(self).__name__)
        self.addParams()
        self.parseParams(config)
        try:
            self.queue = FilterFactory.create_object(
                self.data_queue.keys()[0],
                self.data_queue.values()[0])
            self.queue.init()
        except Exception as ex:
            self.log.error(
                'Unable to create input queue in test "{}"'.format(self.name))
            self.log.error(ex)
            exit(-1)

        self.evaluators = Creator.parse_filters(self.evaluators)

    def set_net(self, net):
        self.log.info(' Getting first batch in test {}'.format(self.name))
        self.net = net
        self.batch = fetch_batch(self.queue, self.batch_size)
        self.log.info(' DONE Getting first batch in test {}'.format(self.name))

    def crop(self, data1, data2):
        b = [data1.shape[2] - data2.shape[2], data1.shape[3] - data2.shape[3]]
        for i in range(len(b)):
            if int(b[i]) % 2:
                self.log.error(' Cant crop network inputs. Sizes are {} and {}. The difference cant be odd.'.format(data1.shape, data2.shape))
                exit(-1)
            b[i] = int(b[i]) / 2

        return data1[:, :, b[0]:-b[0], b[1]:-b[1]]

    def test(self, iteration=0):
        self.log.info(' Running test {}'.format(self.name))
        matched, unmatched = set_shapes(self.net, self.batch)
        self.log.info(' Matched packets to network layers {}'.format(matched))
        self.log.info(' Unmatched packets {}'.format(unmatched))
        loss = defaultdict(list)
        for i in range(self.iterations):
            data = fetch_batch(self.queue, self.batch_size)
            fill_net_input(self.net, data)
            outputs = self.net.forward()
            for output in outputs:
                loss[output].append(self.net.blobs[output].data.mean())

            if self.evaluators:
                gt = self.net.blobs[self.gt_layer].data
                result = self.net.blobs[self.result_layer].data
                original = self.net.blobs[self.data_layer].data
                gt = self.crop(gt, result)
                original = self.crop(original, result)
                if self.display:
                    tmp = cv2.resize(
                        gt[0, :, :, :].transpose(1, 2, 0) + 0.5, (0, 0),
                        fx=2, fy=2)
                    cv2.imshow(self.name + ' gt', tmp)
                    tmp = cv2.resize(
                        result[0, :, :, :].transpose(1, 2, 0) + 0.5, (0, 0),
                        fx=2, fy=2)
                    cv2.imshow(self.name + ' result', tmp)
                    tmp = cv2.resize(
                        original[0, :, :, :].transpose(1, 2, 0) + 0.5, (0, 0),
                        fx=2, fy=2)
                    cv2.imshow(self.name + ' original', tmp)
                    cv2.waitKey(20)

                if self.write:
                    cv2.imwrite(
                        'output_{:07d}_{}_gt.png'.format(iteration, i),
                        gt[0, :, :, :].transpose(1, 2, 0) *256 + 126)
                    cv2.imwrite(
                        'output_{:07d}_{}_result.jpg'.format(iteration, i),
                        result[0, :, :, :].transpose(1, 2, 0) *256 + 126)
                    cv2.imwrite(
                        'output_{:07d}_{}_original.png'.format(iteration, i),
                        original[0, :, :, :].transpose(1, 2, 0) *256 + 126)

                for evaluator in self.evaluators:
                    evaluator.add(gt=gt, original=original, result=result)

        for layer in loss:
            value = np.average(loss[layer])
            self.log.info(' Iteration {} test {} metric loss {}: {}'.format(
                iteration, self.name, layer, value))

        for evaluator in self.evaluators:
            results = evaluator.getResults()
            for metric in results:
                self.log.info(' Iteration {} test {} metric {} : {} {} {}'.format(
                    iteration, self.name, metric,
                    results[metric][1], results[metric][0],
                    results[metric][0] - results[metric][1]))
            evaluator.clear()
        self.log.info(' DONE Test {}'.format(self.name))


class filterNormalizer(object):
    def __init__(self, layers):
        self.layers = layers

    def normalize(self, net):
        for layer in net.params:
            if layer in self.layers:
                dim = len(net.params[layer][0].data.shape)
                data = net.params[layer][0].data
                data = data.reshape(data.shape[0], -1)
                mean = np.average(data, axis=1).reshape(
                    [-1] + [1] * (dim - 1))

                data = net.params[layer][0].data - mean
                energy = np.sum(
                    data.reshape(data.shape[0], -1)**2, axis=1).reshape(
                    [-1] + [1] * (dim - 1))
                net.params[layer][0].data[...] = data / energy


class batchBuffer(object):
    def __init__(self):
        self.buffer = []
        self.costs = []
        self.max_size = 2000

    def add(self, batch):
        self.buffer = [batch] + self.buffer
        self.costs = [100.0] + self.costs

        if len(self.buffer) >= self.max_size:
            self.buffer.pop()
            self.costs.pop()

        return 0

    def get(self):
        #prob = np.asarray(self.costs)
        #prob = prob / prob.sum()
        id = np.random.choice(len(self.buffer))
        return self.buffer[id], id, 0

    def updateCost(self, id, cost):
        self.costs[id] = cost


class Trainer(Configurable, multiprocessing.Process):
    def addParams(self):
        self.params.append(parameter(
            'batch_size', required=True,
            parser=lambda x: max(int(x), 1),
            help='Size of training and testing minibatches.'))
        self.params.append(parameter(
            'max_iter', required=False, default=100000,
            parser=lambda x: max(int(x), 1),
            help='Maximum number of training iterations'))
        self.params.append(parameter(
            'use_gpu', required=False, default=True,
            parser=bool,
            help='Should GPU be used?'))
        self.params.append(parameter(
            'gpu_id', required=False, default=0,
            parser=lambda x: max(int(x), 0),
            help='GPU which should be used.'))
        self.params.append(parameter(
            'caffe_solver_file', required=True,
            help='Caffe solver configuration.'))
        self.params.append(parameter(
            'caffe_solver_state', required=False,
            help='Caffe solver state from which should training continue.'))
        self.params.append(parameter(
            'caffe_weights', required=False,
            help='Caffe model file used to initialize weights.'))
        self.params.append(parameter(
            'test_interval', required=False, default=1000,
            parser=lambda x: max(int(x), 1),
            help='How often should tests be run?'))
        self.params.append(parameter(
            'stat_interval', required=False, default=100,
            parser=lambda x: max(int(x), 1),
            help='How often should network statistics be computed.'))
        self.params.append(parameter(
            'train_data', required=True,
            parser=lambda x: x,
            help='Queue from which data should be read.'))
        self.params.append(parameter(
            'tests', required=False, default=[],
            parser=lambda x: x,
            help='List of test specifications.'))
        self.params.append(parameter(
            'save_filters', required=False, default=None,
            parser=bool,
            help='Will save collage of first layer filters.'))
        self.params.append(parameter(
            'normalize_layers', required=False, default=[],
            parser=list,
            help='List of layer names which should have filters normalized.'))


    def __init__(self, config):
        multiprocessing.Process.__init__(self)
        Configurable.__init__(self)
        #super(Trainer, self).__init__()
        self.daemon = True
        self.log = logging.getLogger(__name__ + "." + type(self).__name__)
        self.addParams()
        self.parseParams(config)

        self.stat = None

        self.testers = []
        for test in self.tests:
            self.testers.append(Tester(test))

        self.train_in_queue = FilterFactory.create_object(
            self.train_data.keys()[0], self.train_data.values()[0])
        self.train_in_queue.init()
        self.batchQueue = Queue.Queue(maxsize=2)
        self.buffer = batchBuffer()

        self.layerNormalizer = filterNormalizer(self.normalize_layers)

    def center_initialization(self, net, step=1):
        """
        Center the net parameters - subtract the non-zero mean
        """
        for layer in net.params:
            if len(net.params[layer][0].data.shape) == 4:
                params = net.params[layer][0]
                average = np.average(params.data, (1, 2, 3))
                # pylint: disable=E1101
                average = average.reshape((-1, 1, 1, 1))
                params.data[...] = params.data - (average * step)
        self.log.info("Net - centered parameters.")

    def init_caffe(self, solver_file):
        """
        Initialize the caffe solver.
        """
        if self.use_gpu:
            self.log.info(" Using GPU id {}".format(self.gpu_id))
            caffe.set_device(self.gpu_id)
            caffe.set_mode_gpu()
        else:
            self.log.info(" Using CPU")
            caffe.set_mode_cpu()

        self.log.info(" Reading solver file: %s", solver_file)
        solver = caffe.get_solver(solver_file)

        if self.caffe_solver_state:
            self.log.info(" Loading solver state: {}".format(self.caffe_solver_state))
            solver.restore(self.caffe_solver_state)
        elif self.caffe_weights:
            self.log.info(" Loading weights: %s", self.caffe_weights)
            solver.net.copy_from(self.caffe_weights)
        else:
            self.log.info(" Centering network parameters.")
            self.center_initialization(solver.net)
        return solver

    def train(self, solver):
        """
        Run one solver step
        """
        time1 = time.clock()
        if not self.batchQueue.empty():
            data = self.batchQueue.get()
            batch_id = self.buffer.add(data)
        else:
            data, batch_id, prob = self.buffer.get()
            from time import sleep
            sleep(0.002)
            #self.log.info(
            #    ' Reusing training minibatch {}, prob {}. Iteration {}.'.format(batch_id, prob, solver.iter))
        time2 = time.clock()
        fill_net_input(solver.net, data)

        time3 = time.clock()
        solver.step(1)
        self.layerNormalizer.normalize(solver.net)
        self.buffer.updateCost(batch_id, solver.net.blobs['loss'].data)
        time4 = time.clock()
        self.log.debug(" Train time (fetch, fill, step): {:f} {:f} {:f} {:f}".format(time2-time1, time3-time2, time4-time3, time4-time1))

    def fetchThread(self):
        while True:
            data = fetch_batch(self.train_in_queue, self.batch_size)
            self.batchQueue.put(data)


    def run(self):
        """
        Run the training
        """

        self.log.info(' Initializing caffe')
        solver = self.init_caffe(self.caffe_solver_file)
        self.log.info(' Reading initial batch from training data queue. (May take a while if image buffers are used in data pipeline.')
        batch = fetch_batch(self.train_in_queue, self.batch_size)
        for name in batch:
            self.log.info(' Training input layer shape: {} - {}'.format(
                name, batch[name].shape))
        self.log.info(' Resizing training network.')
        matched, unmatched = set_shapes(solver.net, batch)
        self.log.info(' Matched packets to network layers {}'.format(matched))
        self.log.info(' Unmatched packets {}'.format(unmatched))
        if unmatched:
            self.log.error(' Some packets do not match to network layers: {}'.format(unmatched))
            exit(-1)

        self.log.info(' Setting up test networks.')
        for testId, tester in enumerate(self.testers):
            tester.set_net(solver.test_nets[0])

        self.log.info(' Setting up network statistics collector.')
        self.stat = ActivationSimpleStat(solver.net)

        self.log.info(' Starting data fetching thread.')

        self.fetchThreadObj = threading.Thread(target=self.fetchThread,)
        self.fetchThreadObj.setDaemon(True)
        self.fetchThreadObj.start()

        self.buffer.add(self.batchQueue.get())

        self.log.info(' Running training.')
        while solver.iter < self.max_iter:
            # Training
            start_time = time.clock()
            self.train(solver)
            self.log.debug(
                "Train Iteration time: {}".format(time.clock() - start_time))

            # Testing
            if solver.iter % self.test_interval == 0:
                for tester in self.testers:
                    start_time = time.clock()
                    tester.test(iteration=solver.iter)
                    self.log.debug(" Test Iteration time: {}".format(
                        time.clock() - start_time))

            # Print and save stats
            if solver.iter % self.stat_interval == 0:
                #self.stat.add_history(solver.net)
                self.stat.print_stats(solver.net, solver.iter)
                self.stat.filter_stats(solver.net, solver.iter)

                #data = solver.net.blobs['data'].data[...].transpose(0,2,3,1)
                #data = data.reshape(-1, data.shape[2], data.shape[3])
                #dataOut = solver.net.blobs['cout-scale'].data[...].transpose(0,2,3,1)
                #dataOut = dataOut.reshape(-1, dataOut.shape[2], dataOut.shape[3])
                #cv2.imwrite('{:06d}_data.jpg'.format(solver.iter), data + 127)
                #cv2.imwrite('{:06d}_dataOut.jpg'.format(solver.iter), dataOut + 127)

            self.log.debug(" Iteration: %i", solver.iter)

        self.log.info(" Training DONE.")

    def test(self, net, in_queue):
        """
        Run all the defined test networks
        """
        self.log.info(" Start phase Test.")
        for _ in xrange(self.test_iter):
            fetch_flag = self.fetch_batch(in_queue, net)
            net.forward()

        self.log.info(" End phase Test.")
        return fetch_flag


class ActivationSimpleStat(object):
    """
    Compute the activation stats
    """

    bins = [-10000, 0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9,
            0.99, 10000]

    def __init__(self, net, history_size=20):
        """
        Constructor of actiavation stats
        """
        learn_param_keys = [key for key in net.blobs if 'split' not in key]
        # in net.params if key in net.blobs]
        self.list_blobs = learn_param_keys
        self.log = logging.getLogger(__name__ + ".Stats")
        self.binCount = 100

        self.startBlobs = self.getBlobDict(net)
        self.lastBlobs = self.getBlobDict(net)

    def getBlobDict(self, net):
        blobs = {}
        for layer in net.params:
            for rank, blob in enumerate(net.params[layer]):
                if len(blob.data.shape) in [2, 4]:
                    blobs[(layer, rank)] = np.copy(blob.data.reshape(blob.data.shape[0], -1))
        return blobs

    def print_stats(self, net, iteration):
        """
        Add the layers activations to compute the stats
        """

        for key in self.list_blobs:
            # average only of positive values
            # average of every activation map in the batch

            if len(net.blobs[key].shape) == 2:
                data = net.blobs[key].data.transpose(1, 0)
            elif len(net.blobs[key].shape) == 4:
                data = net.blobs[key].data.transpose(1, 0, 2, 3)
                data = data.reshape((data.shape[0], -1))
            else:
                continue

            # avg_activation = np.average(data > 0, (1))

            bins = np.arange(self.binCount+1).astype(np.float32) / self.binCount * (data.max()-data.min()) + data.min()
            qdata = ((data - data.min()) / (data.max() - data.min()) * self.binCount + 0.5).astype(np.int32)
            self.log.info('{} - {}'.format(key, (qdata < 0).sum()))
            histograms = [np.bincount(line, minlength=self.binCount+1)
                          for line in qdata]
            fig = plt.figure()
            fig.set_size_inches(20, 11)
            ax = fig.add_subplot(1, 1, 1)
            fig.subplots_adjust(left=0.02, right=0.99, bottom=0.03, top=0.99)
            for i in histograms:
                cum = np.cumsum(i).astype(np.float)
                ax.plot(bins, cum / cum.max())
            fig.savefig('{}_{:07d}.png'.format(key, iteration))
            plt.close(fig)

    def blobCorrelation(self, data, data2):
        sim = []
        for x, y in zip(data, data2):
            sim.append(
                1.0 - x.dot(y.T) / (x.dot(x)**0.5 * y.dot(y)**0.5))
        return sim

    def filter_stats(self, net, iteration):
        for layer in net.params:
            for rank, blob in enumerate(net.params[layer]):
                if len(blob.data.shape) in [2, 4]:
                    data = blob.data.reshape(blob.data.shape[0], -1)

                    startSim = self.blobCorrelation(
                        data, self.startBlobs[(layer, rank)])
                    plt.hist(startSim, bins=40, range=(-1.0, 1.0),
                             normed=1, facecolor='green')
                    fig = plt.gcf()
                    fig.savefig(
                        'sim_{}_{}_{:07d}.png'.format(layer, rank, iteration))
                    plt.close(fig)

                    lastSim = self.blobCorrelation(
                        data, self.lastBlobs[(layer, rank)])

                    fig = plt.figure()
                    ax = fig.add_subplot(1, 1, 1)
                    ax.plot(startSim, lastSim, 'b+')
                    ax.set_ylabel('since last')
                    ax.set_xlabel('since start')
                    fig.savefig('sim_last_{}_{}_{:07d}.png'.format(
                        layer, rank, iteration))
                    plt.close(fig)

                if len(blob.data.shape) == 4 and (blob.data.shape[1] == 3 or blob.data.shape[1] == 1):
                    filters = [f for f in blob.data.transpose(0, 2, 3, 1)]
                    # filters = [(x / (np.absolute(x).max()) + 0.5) * 256
                    #           for x in filters]
                    side = int(np.ceil(len(filters)**0.5))
                    for i in range(side**2 - len(filters)):
                        filters.append(filters[-1])
                    collage = [np.concatenate(filters[i::side], axis=0)
                               for i in range(side)]
                    collage = np.concatenate(collage, axis=1)
                    collage = (collage / (np.absolute(collage).max()) +
                               0.5) * 256
                    cv2.imwrite(
                        'filter_{}_{}_{:07d}.png'.format(layer, rank, iteration),
                        collage)
        self.lastBlobs = self.getBlobDict(net)



    def xcprint_stats(self):
        """
        Print column aligned activation stats
        """
        format_msg = []
        for key, data in self.list_blobs:
            avg = sum(data) / data.size

            hist, bins = np.histogram(avg, self.bins)
            hist = hist * (1.0 / np.sum(hist))
            msg_list = ["{0:3}".format(int(val * 100 + 0.5)) for val in hist]
            msg_list.insert(0, key + ":")
            format_msg.append(msg_list)
        widths = [max(map(len, col)) for col in zip(*format_msg)]
        for row in format_msg:
            msg = "  ".join((val.ljust(width) for val, width in
                             zip(row, widths)))
            self.log.info(msg)


class ActivationStat(object):
    """
    Compute the activation stats
    """

    bins = [-10000, 0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9,
                    0.99, 10000]

    def __init__(self, net, history_size=20):
        """
        Constructor of actiavation stats
        """
        self.history_size = history_size
        # list of tuples (blobs_name, blob_data) related to learnable params
#         learn_param_keys = net.params.viewkeys() & net.blobs.viewkeys()
        learn_param_keys = [key for key in net.params if key in net.blobs]
        self.list_blobs = [(key, RoundBuffer(history_size))
                           for key in learn_param_keys]
        # dict of params
#         self.dict_params = {key: RoundBuffer(history_size)
#                       for key in net.params.keys()}
        self.log = logging.getLogger(__name__ + ".Stats")

    def add_history(self, net):
        """
        Add the layers activations to compute the stats
        """
        for key, data in self.list_blobs:
            # average only of positive values
            # average of every activation map in the batch
            if len(net.blobs[key].shape) == 2:
                avg_data = np.average(net.blobs[key].data > 0, (0))
            elif len(net.blobs[key].shape) == 4:
                avg_data = np.average(net.blobs[key].data > 0, (0, 2, 3))

            data.append_round(avg_data)

    def print_stats(self):
        """
        Print column aligned activation stats
        """
        format_msg = []
        for key, data in self.list_blobs:
            avg = sum(data) / data.size

            hist, bins = np.histogram(avg, self.bins)
            hist = hist * (1.0 / np.sum(hist))
            msg_list = ["{0:3}".format(int(val * 100 + 0.5)) for val in hist]
            msg_list.insert(0, key + ":")
            format_msg.append(msg_list)
        widths = [max(map(len, col)) for col in zip(*format_msg)]
        for row in format_msg:
            msg = "  ".join((val.ljust(width) for val, width in
                             zip(row, widths)))
            self.log.info(msg)
