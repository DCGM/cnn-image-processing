'''
Created on May 27, 2016

@author: isvoboda
'''

from __future__ import print_function
from __future__ import division

import logging
from matplotlib import pyplot as plt
# pylint: disable=import-error,no-name-in-module
from distutils.util import strtobool
import numpy as np
import caffe
from ..utils import RoundBuffer


class PyPSNRL(caffe.Layer):

    """
    Compute a PSNR of two inputs and PSNR with IPSNR of 3 inputs
    Args:
        max: float
            The maximum value in the input data used to compute the PSNR.
            Default is 255
        history_size: uint
            Size of history a floating avarge is computed from.
            Default is 50
        print_step: uint
            Ith iteration to print the PSNR
            Default is 50
        average: str
            'mse' (default) computes the psnr of average per minibatch
            'psnr': compute the average of psnr per patches
        graph: boolean
            True: plot the graph
            False (default) do not plot
        graph_name: str
            name of the plotted graph
    """

    def setup(self, bottom, top):
        """
        Setup the layer
        """
        self.log = logging.getLogger(__name__ + ".PyPSNRL")
        if len(bottom) < 2 or len(bottom) > 3:
            raise Exception("Need two inputs at least or 3 at most.")
        self.dict_param = dict((key.strip(), val.strip()) for key, val in
                               (item.split(':') for item in
                                self.param_str.split(',')))
        if 'max' in self.dict_param:
            self.max = float(self.dict_param['max'])
        else:
            self.max = 255.
        if 'history_size' in self.dict_param:
            self.history_size = int(self.dict_param['history_size'])
        else:
            self.history_size = 50
        if 'print_step' in self.dict_param:
            self.print_step = int(self.dict_param['print_step'])
        else:
            self.print_step = 50
        if 'graph' in self.dict_param:
            self.plot_graph = strtobool(self.dict_param['graph'])
        else:
            self.plot_graph = False
        if 'graph_name' in self.dict_param:
            self.graph_name = self.dict_param['graph_name']
        else:
            self.graph_name = 'iPSNR.png'

        if 'average' in self.dict_param:
            assert ('mse' == self.dict_param['average'] or
                    'psnr' == self.dict_param['average'])
            self.average = self.dict_param['average']
        else:
            self.average = 'mse'

        self.iterations = 0
        self.psnr_buffers = [RoundBuffer(max_size=self.history_size)
                             for _ in xrange(len(bottom) - 1)]
        self.history = []
        if self.plot_graph:
            self.fig = plt.figure()
            self.axe = self.fig.add_subplot(111)

    def reshape(self, bottom, top):
        """
        Reshape the activation - data blobs
        """
        if len(top) > len(bottom):
            raise Exception("Layer produce more outputs then has its inputs.")

        for i_input in xrange(len(top)):
            top[i_input].reshape(*bottom[i_input].data.shape)

    def forward(self, bottom, top):
        """
        Feed forward
        """
        psnr_list = None

        if self.average == 'psnr':
            psnr_list = self.psnr_patches(bottom)
            for i_val, val in enumerate(psnr_list):
                for psnr_val in val:
                    self.psnr_buffers[i_val].append_round(psnr_val)
        else:
            psnr_list = self.psnr_minibatch(bottom)
            for i_val, psnr_val in enumerate(psnr_list):
                self.psnr_buffers[i_val].append_round(psnr_val)

        if self.iterations % self.print_step == 0:
            avg_psnr = [sum(val) / val.size for val in self.psnr_buffers]
            msg = " ".join(' PSNR bottom[{}]: {}'
                           .format(*val) for val in enumerate(avg_psnr))
            if len(psnr_list) < 2:
                self.log.info(msg)
            elif len(psnr_list) == 2:
                ipsnr = avg_psnr[0] - avg_psnr[1]
                ipsnr_msg = " iPSNR: {} ".format(ipsnr)
                self.log.info(" ".join([ipsnr_msg, msg]))

                if self.plot_graph:
                    self.history.append(ipsnr)
                    self.plot()

        for i_data in xrange(len(top)):
            top[i_data].data[...] = bottom[i_data].data

        self.iterations += 1

    def backward(self, top, propagate_down, bottom):
        """
        Layer bacpropagation
        """
        for i_diff in xrange(len(top)):
            bottom[i_diff].diff[...] = top[i_diff].diff

    def psnr_patches(self, bottom):
        """
        Compute the average PSNR of bottom per axis 1 (bottom[0,0-end])
        """
        results = []
        for i_input in xrange(len(bottom) - 1):
            diff = (bottom[-1].data - bottom[i_input].data).astype(np.float64)
            bottom_psnr = []
            for i_img in xrange(diff.shape[0]):
                ssd = (diff[i_img] ** 2).sum()
                mse = ssd / float(diff[i_img].size)
                if mse == 0:
                    bottom_psnr.append(np.nan)
                else:
                    psnr = 10 * np.log10(self.max ** 2 / mse)
                    bottom_psnr.append(psnr)
            results.append(bottom_psnr)
        return results

    def psnr_minibatch(self, bottom):
        '''
        Compute PSNR of the whole minibatch
        '''
        list_bottom_psnr = []
        for i_input in xrange(len(bottom) - 1):
            diff = (bottom[-1].data - bottom[i_input].data).astype(np.float64)
            ssd = (diff ** 2).sum()
            mse = ssd / float(diff.size)
            if mse == 0:
                list_bottom_psnr.append(np.nan)
            else:
                psnr = 10 * np.log10(self.max ** 2 / mse)
                list_bottom_psnr.append(psnr)
        return list_bottom_psnr

    def plot(self):
        """
        Plot the iPSNR history to graph
        """
        self.axe.cla()
        self.axe.plot(self.history)
        self.axe.set_xlabel("Iteration / {}".format(self.print_step))
        self.axe.set_ylabel("iPSNR")
        self.fig.savefig(self.graph_name)
