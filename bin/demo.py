#!/usr/bin/env python
from __future__ import print_function

import sys
import caffe
import numpy as np
import cv2

def parseArgs():
    import argparse
    parser = argparse.ArgumentParser(description="Run network on imput from camera.")
    parser.add_argument("-d", "--net-definition", 
                        required=True, help="Network definition")

    parser.add_argument("-m", "--model", 
                        required=True, help="Caffe model file")

    parser.add_argument("-v", "--verbose", action="store_true",
                        help="Set the verbose mode.")

    parser.add_argument('-c', '--crop-resolution',
                        default=400,
                        type=int,
                        help='Crop center of video frames.')

    parser.add_argument('--input-layer',
                        default='data',
                        help='Name of input layer.')

    parser.add_argument('--output-layer',
                        default='out',
                        help='Name of output layer.')

    args = parser.parse_args()

    for key, val in vars(args).iteritems():
        print("{}: {}".format(key, val))

    return args


class network(object):
    def __init__(self, definitionFile, modelFile, 
                 inputlayer, outputLayer, resolution):
        self.net = caffe.Net(definitionFile, modelFile, caffe.TEST)
        self.inputlayer = inputlayer
        self.outputLayer = outputLayer
        self.resolution = resolution

        self.net.blobs[self.inputlayer].reshape(1, 3, resolution, resolution)
        self.net.reshape()

    def process(self, img):
        img = img.astype(np.float32) / 256.0 - 0.5
        img = img.transpose(2,0,1)
        self.net.blobs[self.inputlayer].data[0] = img
        self.net.forward()
        output = self.net.blobs[self.outputLayer].data[0].transpose(1,2,0)
        output += 0.5
        return output 

def centralCrop(img, resolution):
    border = (int((img.shape[0] - resolution) / 2),
              int((img.shape[1] - resolution) / 2))
    outImg = img[
        border[0]:border[0] + resolution,
        border[1]:border[1] + resolution]  
    return outImg 

def main():
    # Print the arguments
    args = parseArgs()

    caffe.set_device(0)
    caffe.set_mode_gpu()

    net = network(args.net_definition, args.model, 
                  args.input_layer, args.output_layer, 
                  args.crop_resolution)

    cap = cv2.VideoCapture(0)

    while(cap.isOpened()):
        # Capture frame-by-frame
        ret, frame = cap.read()
        if not ret:
            break

        crop = centralCrop(frame, args.crop_resolution)
        processedCrop = net.process(crop)

        # Display the resulting frame
        cv2.imshow('input', crop)
        cv2.imshow('output', processedCrop)
        key = cv2.waitKey(5) & 0xFF
        if key == ord('q') or key == ord('x') or key == 27:
            break

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()

 
if __name__ == "__main__":
    main()
