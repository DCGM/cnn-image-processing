#!/usr/bin/env python

from __future__ import print_function

import sys
import caffe
import cPickle as pickle
import sys
from collections import defaultdict
import numpy as np
from matplotlib import pyplot as plt


def parse_args():
    print(' '.join(sys.argv))
    import argparse
    parser = argparse.ArgumentParser(description="Eval the cnn")
    parser.add_argument("-p", "--pickle-file", required=True,
                        help="Pickle file with data.")
    parser.add_argument("-a", "--annotation", required=False,
                        help="Annotation file.")
    parser.add_argument("-c", "--classifier",
                        required=False, action="store_true",
                        help="True if the network is a classifier.")
    args = parser.parse_args()
    return args


def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()


def classToFloat(res):
    rangeVal = np.arange(res.shape[1]) / 90.0 * 3.14 - 1.5708
    rangeVal = rangeVal.reshape(1, -1)
    for x in res:
        x[...] = softmax(x)
    results = (res * rangeVal).sum(axis=1)
    return results.reshape(-1, 1)


def readAnnotation(fileName):
    ann = {}
    with open(fileName, 'r') as f:
        for line in f:
            w = line.strip().split()
            values = np.asarray([float(x) for x in w[1].split(',')])
            ann[w[0]] = values
    return ann


def main():
    args = parse_args()

    data = pickle.load(open(args.pickle_file, 'r'))

    caffe.set_device(0)
    caffe.set_mode_gpu()


    for line in sys.stdin:
        results = defaultdict(list)
        identifier, model, deploy = line.split()

        net = caffe.Net(deploy, model, caffe.TEST)
        batchSize = net.blobs['data'].data.shape[0]

        with open(identifier + '.res', 'w') as f:
            for i in range(0, len(data), batchSize):
                paths = []
                for j in range(batchSize):
                    if i + j >= len(data):
                        break
                    paths.append(data[i + j][0]['path'])
                    for packet in data[i + j]:
                        if packet['label'] in net.blobs:
                            net.blobs[
                                packet['label']].data[j, :, :, :] = packet['data']

                net.forward()
                if args.classifier:
                    yaw = classToFloat(net.blobs['out_yaw'].data)
                    pitch = classToFloat(net.blobs['out_pitch'].data)
                    output = np.concatenate([yaw, pitch], axis=1)
                else:
                    output = net.blobs['out'].data

                for path, res in zip(paths, output):
                    res = res.reshape(-1)
                    results[path].append(res.reshape(1, -1).copy())
                    print(path, '{},{}'.format(res[0], res[1]), file=f)

        if args.annotation:
            ann = readAnnotation(args.annotation)
            error = np.zeros(2)
            for path in results:
                res = results[path]
                res = np.concatenate(res, axis=0)
                res = res.mean(axis=0)
                error += np.absolute(res - ann[path])

            print(identifier, error / len(results))


if __name__ == "__main__":
    main()
