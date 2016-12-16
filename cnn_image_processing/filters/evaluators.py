from __future__ import division
from __future__ import print_function

import logging
import numpy as np

from ..utilities import parameter, Configurable


class PSNR(Configurable):
    def addParams(self):
        self.params.append(parameter(
            'max_value', required=False, default=255,
            parser=lambda x: max(float(x), 0),
            help='The maximum input value used to compute the PSNR.'))
        self.params.append(parameter(
            'full_batch_compute', required=False, default=False,
            parser=bool,
            help='Should psnr be computed over whole batches.'))

    def __init__(self, config):
        super(PSNR, self).__init__()
        self.log = logging.getLogger(__name__ + "." + type(self).__name__)
        self.addParams()
        self.parseParams(config)
        self.clear()

    def clear(self):
        self.mse = []
        self.psnr = []
        self.mse_orig = []
        self.psnr_orig = []

    def computeError(self, data, gt):
        diff2 = (gt - data)**2
        if self.full_batch_compute:
            mse = np.mean(diff2)
        else:
            axis = tuple(range(1, len(data.shape)))
            mse = np.mean(diff2, axis=axis)

        psnr = np.mean(10 * np.log10(self.max_value**2 / (mse + (self.max_value*0.00001)**2)))
        mse = np.mean(mse)
        return mse, psnr

    def add(self, gt, result, original=None):
        mse, psnr = self.computeError(gt, result)
        self.mse.append(mse)
        self.psnr.append(psnr)
        if original is not None:
            mse_orig, psnr_orig = self.computeError(gt, original)
            self.mse_orig.append(mse_orig)
            self.psnr_orig.append(psnr_orig)

    def getResults(self):
        mse = np.mean(self.mse)
        mse_orig = np.mean(self.mse_orig)
        psnr = np.mean(self.psnr)
        psnr_orig = np.mean(self.psnr_orig)
        return {'mse': (mse, mse_orig), 'psnr': (psnr, psnr_orig)}
