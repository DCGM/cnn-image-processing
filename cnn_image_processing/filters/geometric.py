from __future__ import division
from __future__ import print_function

import functools
import logging
import cv2
import numpy as np
import random
from math import sin, cos, asin, atan2, pi
from PIL import Image

from ..utilities import parameter, Configurable, ContinuePipeline, TerminatePipeline


def rotateVectorToYP(vector, rotationMatrix):
    [x2,y2,z2] = np.dot(rotationMatrix, vector.reshape(-1))
    yaw = atan2(x2, -z2)
    pitch = asin(-y2);
    return np.asarray([yaw, pitch]).reshape(1,1,-1)

# class rotateEye(Configurable):
class RotateImage(Configurable):
    """
    Rotates image randomly around it's center and creates
    a function wchich can adequatly rotate gaze vectors.
    This is specific to gaze estimation training.

    Example:
    RotateImage: {roation_range: 10}
    """
    def addParams(self):
        self.params.append(parameter(
            'rotation_range', required=False, default=10, parser=float,
            help='Rotates image around its center by at most rotation_range degrees.'))


    def __init__(self, config):
        Configurable.__init__(self)
        self.log = logging.getLogger(__name__ + "." + type(self).__name__)
        self.addParams()
        self.parseParams(config)

    def rotate_image(self, img, roll_radians):
        #imgData = np.asarray(Image.open(os.path.join("/home/ipavelkova/UnityEyes/imgs_cropped/", "1000.jpg")))
        shiftBackMatrix = np.array(
            [[1, 0, img.shape[1] / 2],
            [0, 1, img.shape[0] / 2],
            [0, 0, 1]])
        shiftMatrix = np.array(
            [[1, 0, -img.shape[1] / 2],
            [0, 1, -img.shape[0] / 2],
            [0, 0, 1]])
        rotationMatrix = np.array([[cos(roll_radians), -sin(roll_radians), 0],
                         [sin(roll_radians), cos(roll_radians),0],
                         [0,0,1]
                         ])
        M = shiftBackMatrix.dot( rotationMatrix.dot(shiftMatrix))

        img = cv2.warpAffine( img, M[0:2], (img.shape[1], img.shape[0])).reshape(img.shape[0], img.shape[1], -1)
        return img


    def __call__(self, packet, previous):
        roll = random.uniform(-self.rotation_range, self.rotation_range)
        roll_radians = roll * pi / 180.
        packet['data'] = self.rotate_image(packet['data'], roll_radians)
        rotationMatrix = np.array([[cos(roll_radians), -sin(roll_radians), 0],
                         [sin(roll_radians), cos(roll_radians),0],
                         [0,0, 1]])

        previous['op'] = functools.partial(rotateVectorToYP, rotationMatrix=rotationMatrix)
        return [packet]
