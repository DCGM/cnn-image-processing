from __future__ import division, print_function
import caffe
import numpy as np
import cv2


class CroppedElementwiseLayer(caffe.Layer):
    """
    Crop input blobs as needed to match the dimension of the smallest blob and perform a element-wise operation. Accepts multiple bottom blobs and outputs a single blob. The input blobs must have the same number of channels.
    """

    def setup(self, bottom, top):
        # check input pair
        if len(bottom) < 2:
            raise Exception("Needs at least two bottom blobs.")
        if not(self.param_str == 'SUM'):
            raise Exception("param_str has to be one of 'SUM'. Got '%s'." % self.self.param_str)

        self.operationStr = self.param_str

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

        # output has the size of minimum input
        top[0].reshape(*self.shape)

    def crop(self, blob):
        crop = blob
        b2 = (crop.shape[2]-self.shape[2])/2
        if b2 > 0:
            crop = crop[:,:, b2:-b2, :]
        b3 = (crop.shape[3]-self.shape[3])/2
        if b3 > 0:
            crop = crop[:,:,:, b3:-b3]

        return crop

    def forward(self, bottom, top):
        # crop bottom[1] to match bottom[0]
        top[0].data[...] = self.crop(bottom[0].data)

        for bID in range( 1, len(bottom)):
            if self.operationStr == 'SUM':
                top[0].data[...] = top[0].data + self.crop(bottom[bID].data)

    def backward(self, top, propagate_down, bottom):
        for bID in range( len(bottom)):
            if propagate_down[bID] and self.operationStr == 'SUM':
                b2 = (bottom[bID].data.shape[2]-self.shape[2])/2
                b3 = (bottom[bID].data.shape[3]-self.shape[3])/2
                bottom[bID].diff[:, :, b2:b2+self.shape[2], b3:b3+self.shape[3]] = top[0].diff



class CropLayer(caffe.Layer):
    """
    Crop first input blob to match the dimension of the second bottom blob.
    """

    def setup(self, bottom, top):
        # check input pair
        if len(bottom) != 2:
            raise Exception("Needs two bottom blobs.")

    def reshape(self, bottom, top):
        # check input dimPSNRensions match
        self.shape = [x for x in bottom[0].data.shape]
        for bID in range( 1, len(bottom)):
            self.shape[2] = min(self.shape[2], bottom[bID].shape[2])
            self.shape[3] = min(self.shape[3], bottom[bID].shape[3])

        # output has the size of minimum input
        top[0].reshape(*self.shape)

    def crop(self, blob):
        b2 = (blob.shape[2]-self.shape[2])/2
        b3 = (blob.shape[3]-self.shape[3])/2
        return blob[:,:,b2:b2+self.shape[2], b3:b3+self.shape[3]]

    def forward(self, bottom, top):
        # crop bottom[1] to match bottom[0]
        top[0].data[...] = self.crop(bottom[0].data)

    def backward(self, top, propagate_down, bottom):
        if propagate_down[0]:
            b2 = (bottom[0].data.shape[2]-self.shape[2])/2
            b3 = (bottom[0].data.shape[3]-self.shape[3])/2
            bottom[0].diff[:, :, b2:b2+self.shape[2], b3:b3+self.shape[3]] = top[0].diff
