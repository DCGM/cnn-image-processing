from __future__ import division, print_function
import caffe
import numpy as np
import cv2

from legacy_utils import log

class CroppedEuclideanLossVisLayer(caffe.Layer):
    """
    Compute the Euclidean Loss on aligned image in the same manner as the C++ EuclideanLossLayer.
    """

    def setup(self, bottom, top):
        # check input pair
        if len(bottom) < 2:
            raise Exception("Needs two inputs to compute distance. Possibly more to show.")

        self.historySize = 200
        self.PSNR = []
        self.IPSNR = []

        self.options = dict([tuple(s.split(':')) for s in self.param_str.split()])

    def crop(self, blob):
        crop = blob
        b2 = (crop.shape[2]-self.shape[2])/2
        if b2 > 0:
            crop = crop[:,:, b2:-b2, :]
        b3 = (crop.shape[3]-self.shape[3])/2
        if b3 > 0:
            crop = crop[:,:,:, b3:-b3]
        return crop

    def reshape(self, bottom, top):

        # check input dimensions match
        self.shape = [x for x in bottom[0].data.shape]
        for bID in range( 1, len(bottom)):
            if self.shape[0] != bottom[bID].shape[0]:
                raise Exception("Number of images in bottom blobs must match. Got %d and %d" % (self.shape[0], bottom[bID].shape[0]))
            if self.shape[1] != bottom[bID].shape[1]:
                raise Exception("Number of channels in bottom blobs must match. Got %d and %d" % (self.shape[1], bottom[bID].shape[1]))

            self.shape[2] = min(self.shape[2], bottom[bID].shape[2])
            self.shape[3] = min(self.shape[3], bottom[bID].shape[3])

        # size of difference is the same as of the first input
        self.diff = np.zeros(self.shape, dtype=np.float32)
        # loss output is scalar
        top[0].reshape(1)


    def computePSNR(self, diff):
        SSD = np.sum(diff**2)
        MSE = SSD / diff.size
        PSNR = 10 * np.log10( 1.0404 / MSE)
        return PSNR, SSD


    def forward(self, bottom, top):
        if 'show' in self.options:
            for bID in range(len(bottom)):
                img = self.crop(bottom[bID].data)[0,:,:,:].transpose([1, 2, 0])
                cv2.imshow('bottom%d' % bID, cv2.resize( img+0.5, (0,0), fx=3, fy=3))
            cv2.waitKey(int(self.options['show']))

        # crop bottom[1] to match bottom[0]
        croppedTruth = self.crop(bottom[1].data)

        self.diff = (bottom[0].data - croppedTruth).astype( np.float64)
        PSNR, SSD = self.computePSNR( self.diff)
        top[0].data[...] = SSD / bottom[0].num / 2.

        self.PSNR.append(PSNR)
        if len(self.PSNR) > self.historySize:
            self.PSNR = self.PSNR[1:]
        log( "PSNR:",  reduce(lambda x, y: x + y, self.PSNR) / len(self.PSNR))

        if len(bottom) >= 3:
            croppedReference = self.crop(bottom[2].data)
            diffRef = (croppedTruth - croppedReference).astype( np.float64)
            PSNR2, SSD = self.computePSNR(diffRef)
            self.IPSNR.append(PSNR - PSNR2)
            if len(self.IPSNR) > self.historySize:
                self.IPSNR = self.IPSNR[1:]
            log( "IPSNR:",  reduce(lambda x, y: x + y, self.IPSNR) / len(self.IPSNR))


    def backward(self, top, propagate_down, bottom):
        i = 0
        if not propagate_down[i]:
            return
        if i == 0:
            sign = 1
        else:
            sign = -1
        bottom[i].diff[...] = sign * self.diff / bottom[i].num
