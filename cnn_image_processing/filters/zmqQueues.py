from __future__ import division
from __future__ import print_function

import logging
import numpy as np
import cPickle as pickle
import zmq
import copy

from ..utilities import parameter, Configurable, ContinuePipeline, TerminatePipeline


class OutQueueZMQ(Configurable):
    """
    Output pipeline queue using ZMQ PUSH queue. Sends list of packets.

    Example:
    InQueueZMQ: {url: "ipc://tst_images"}
    """
    def addParams(self):
        self.params.append(parameter(
            'url', required=True,
            parser=str, help='The queue address (e.g. ipc://images, tcp://1.2.3.4:5012)'))
        self.params.append(parameter(
            'bind', required=False, default=False,
            parser=bool,
            help='Should this end bind? Only one que should bind the an URL.'))

    def __init__(self, config):
        Configurable.__init__(self)
        self.log = logging.getLogger(__name__ + "." + type(self).__name__)
        self.addParams()
        self.parseParams(config)
        self.skipped = 0
        self.ctx = zmq.Context.instance()
        self.s = self.ctx.socket(zmq.PUSH)
        self.s.set_hwm(40)
        if self.bind:
            self.s.bind(self.url)
        else:
            self.s.connect(self.url)

    def __call__(self, packet, previous):
        if 'packets' in previous:
            packets = []
            packets.extend(previous['packets'])
            packets.append(packet)
        else:
            packets = [packet]

        for packet in packets:
            messages = []

            if 'data' in packets and packet['data'].flags.c_contiguous and packet['data'].size > 15000:
                header = pickle.dumps((packet['data'].dtype, packet['data'].shape))
                data = np.getbuffer(packet['data'])
                packet['data'] = None
                messages += [header, data]

        if messages:
            messages = [pickle.dumps(packets)] + messages
            self.s.send_multipart(messages, copy=False)
        else:
            self.s.send_pyobj(packets)


class InQueueZMQ(Configurable):
    """
    Input pipeline queue using ZMQ PULL queue. Recieves list of packets.

    Example:
    InQueueZMQ: {url: "ipc://tst_images", blocking: False, skip: 9, bind: False}
    """

    def addParams(self):
        self.params.append(parameter(
            'url', required=True,
            parser=str, help='The queue address (e.g. ipc://images, tcp://1.2.3.4:5012)'))
        self.params.append(parameter(
            'blocking', required=False, default=True,
            parser=bool,
            help='Should the read block?'))
        self.params.append(parameter(
            'bind', required=False, default=False,
            parser=bool,
            help='Should this end bind? Only one que should bind the an URL.'))
        self.params.append(parameter(
            'skip', required=False, default=0,
            parser=int,
            help='How many times the queue should just return empty results between actual reads.'))

    def __init__(self, config):
        Configurable.__init__(self)
        self.log = logging.getLogger(__name__ + "." + type(self).__name__)
        self.addParams()
        self.parseParams(config)

        self.skipped = self.skip
        self.ctx = zmq.Context.instance()
        self.s = self.ctx.socket(zmq.PULL)
        self.s.set_hwm(40)
        self.packetCount = 0
        self.flags = 0
        if not self.blocking:
            self.flags = zmq.NOBLOCK

        if self.bind:
            self.s.bind(self.url)
        else:
            self.s.connect(self.url)

    def __call__(self, packet, previous):
        if self.skipped >= self.skip:
            try:
                msg = self.s.recv_multipart(copy=False, flags=self.flags)
                packets = pickle.loads(str(msg[0]))
                msg = msg[1:]
                for packet in packets:
                    if 'data' in packet and packet['data'] is None:
                        header = pickle.loads(str(msg[0]))
                        data = np.frombuffer(msg[1], dtype=header[0])
                        packet['data'] = data.reshape(header[1])
                        msg = msg[2:]
                self.skipped = 0
                self.packetCount = len(packets)
                return packets
            except zmq.Again:
                if self.packetCount < 1:
                    raise ContinuePipeline

        self.skipped += 1
        return [{} for x in range(self.packetCount)]
