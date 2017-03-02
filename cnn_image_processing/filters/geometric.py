from __future__ import division
from __future__ import print_function

import math
import functools
import logging
import cv2
import numpy as np
import random
from math import sin, cos, asin, atan2, pi

from ..utilities import parameter, Configurable


def rotateVectorToYP(vector, rotationMatrix):
    [x2, y2, z2] = np.dot(rotationMatrix, vector.reshape(-1))
    yaw = atan2(x2, -z2)
    pitch = asin(-y2)
    return np.asarray([yaw, pitch]).reshape(1, 1, -1)


class RotateImage(Configurable):
    """
    Rotates image randomly around it's center and creates
    a function which can adequatly rotate gaze vectors.
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
        M = shiftBackMatrix.dot(rotationMatrix.dot(shiftMatrix))
        img = cv2.warpAffine(
            img, M[0:2],
            (img.shape[1], img.shape[0])).reshape(img.shape[0], img.shape[1], -1)
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


def warpPerspective(rx, ry, rz, fov, img, positions=None, shift=(0, 0)):
    s = max(img.shape[0:2])
    rotVec = np.asarray((
        rx * np.pi / 180,
        ry * np.pi / 180,
        rz * np.pi / 180))
    rotMat, j = cv2.Rodrigues(rotVec)
    rotMat[0, 2] = 0
    rotMat[1, 2] = 0
    rotMat[2, 2] = 1

    trnMat1 = np.asarray(
        (1, 0, -img.shape[1] / 2,
         0, 1, -img.shape[0] / 2,
         0, 0, 1)).reshape(3, 3)

    T1 = np.dot(rotMat, trnMat1)
    distance = (s / 2) / math.tan(fov * np.pi / 180)
    T1[2, 2] += distance

    cameraT = np.asarray(
        (distance, 0, img.shape[1] / 2 + shift[1],
         0, distance, img.shape[0] / 2 + shift[0],
         0, 0, 1)).reshape(3, 3)

    T2 = np.dot(cameraT, T1)

    newImage = cv2.warpPerspective(
        img, T2, (img.shape[1], img.shape[0]),
        borderMode=cv2.BORDER_REPLICATE, flags=cv2.INTER_LANCZOS4)
    if positions is None:
        return newImage
    else:
        return newImage, np.squeeze(
            cv2.perspectiveTransform(positions[None, :, :], T2), axis=0)


class VirtualCamera(Configurable):
    """
    Projects the image as a plane using a virtual camera

    Example:
    VirtualCamera: {rotationX_sdev: 10, rotationY_sdev: 10, rotationZ_sdev: 10}
    """
    def addParams(self):
        self.params.append(parameter(
            'rotationX_sdev', required=False, default=10, parser=float,
            help='Std. dev. of rotations around x axis (image axis).'))
        self.params.append(parameter(
            'rotationY_sdev', required=False, default=10, parser=float,
            help='Std. dev. of rotations around x axis (image axis).'))
        self.params.append(parameter(
            'rotationZ_sdev', required=False, default=10, parser=float,
            help='Std. dev. of rotations around z axis (out of plane).'))

    def __init__(self, config):
        Configurable.__init__(self)
        self.log = logging.getLogger(__name__ + "." + type(self).__name__)
        self.addParams()
        self.parseParams(config)

    def __call__(self, packet, previous):
        rx = np.random.standard_normal() * self.rotationX_sdev
        ry = np.random.standard_normal() * self.rotationY_sdev
        rz = np.random.standard_normal() * self.rotationZ_sdev
        fov = np.random.uniform(1, 10)
        if rx != 0 or ry != 0 or rz != 0:
            packet['data'] = warpPerspective(rx, ry, rz, fov, packet['data'])

        return [packet]

