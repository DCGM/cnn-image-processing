# -*- coding: utf-8 -*-
# by Kuba Sochor
from __future__ import division, print_function
import numpy as np
import sys
import os
import time
import platform
import tqdm
import itertools
import re
import caffe
import cv2
import lmdb
import glob
import subprocess
import cPickle
from caffe.proto import caffe_pb2

"""
PROGRESSBAR
"""
class SmartProgressbar:
    def __init__(self, *args, **kwargs):
            self.pbar = tqdm.tqdm(*args, **kwargs)
            self.lastVal = 1

    def update(self, val):
            self.pbar.update(val-self.lastVal)
            self.lastVal = val


    def finish(self):
            self.pbar.close()

def getProgressBar(text, items):
    pbar =  SmartProgressbar(total=items, leave=True, desc=text, file=sys.stdout, unit_scale=True)
    return pbar

def log(*argv, **kwargs):
    print( *argv, file=sys.stderr, **kwargs)

"""
FILE HANDLING
"""
def ensureDir(d):
    if not os.path.exists(d):
        try:
            os.makedirs(d)
        except OSError as e:
            if e.errno != 17: # FILE EXISTS
                raise e

def saveWrapper(filePath, arr):
    ensureDir(os.path.dirname(filePath))
    np.save(filePath, arr)

def replaceTextInFile(filePath, current, replace):
    with open(filePath) as f:
        text = f.read()
    occurences = len(re.findall(re.escape(current), text))
    log("REPLACING [%s] '%s' -> '%s' count: %d"%(filePath, current, replace, occurences))
    text = text.replace(current, replace)
    with open(filePath, "w") as f:
        f.write(text)


def addIfRelative(p, absP):
    if os.path.isabs(p):
        return p
    else:
        return os.path.join(absP, p)

def loadCache(cacheFile):
    with open(cacheFile, 'rb') as fid:
        return cPickle.load(fid)

def saveCache(cacheFile, data):
    ensureDir(os.path.dirname(cacheFile))
    with open(cacheFile, 'wb') as fid:
        cPickle.dump(data, fid, cPickle.HIGHEST_PROTOCOL)

"""
LOGING
"""
def getCurrentTimeString():
    now = time.strftime('%c')
    return "%s" % now


"""
DICTIONARY
"""
def updateDict(d, k, addVal=1):
    if k not in d:
        d[k] = 0
    d[k] += addVal


def getInverseDict(d):
    inverse = {}
    for key, value in d.iteritems():
        assert value not in inverse
        inverse[value] = key
    return inverse


"""
VECTORS
"""
def normalizeVec(v):
    norm=np.linalg.norm(v)
    if norm==0:
        return v
    return v/norm


def softmax(row):
    return np.exp(row)/np.sum(np.exp(row))



def isMonotonic(x):
    dx = np.diff(x)
    return np.all(dx <= 0) or np.all(dx >= 0)

def pointToLineProjection(l, p):
    p = p/p[-1]
    c = p[0]*l[1] - p[1]*l[0]
    perpendicularLine = np.array([-l[1], l[0], c])
    intersection = np.cross(l, perpendicularLine)
    return intersection/intersection[-1]

"""
CAFFE
"""
def getDataFromLMDB(inputLMDB, featVectorSize):
    env = lmdb.open(inputLMDB, readonly = True, map_size = 10 * 1024 * 1024 * 1024)
    featArr = np.zeros((env.stat()["entries"], featVectorSize))
    #print(env.stat())
    pbar = getProgressBar("Transforming features from %s"%inputLMDB, env.stat()["entries"])
    with env.begin() as txn:
        cursor = txn.cursor()
        i = 0
        for key, value in cursor:
            datum = caffe.proto.caffe_pb2.Datum()
            datum.ParseFromString(value)
            data = np.array(datum.float_data).reshape(1, featVectorSize)
            featArr[i,:] = data
            i += 1
            pbar.update(i)
    pbar.finish()
    return featArr


def getLatestSnapshotFile(d):
    files = glob.glob(d + "_*.caffemodel")
    if not files:
        files = glob.glob(d + "/net_*.caffemodel")
    if not files:
        return None
    files.sort(key=lambda x: int(re.findall(r"\d+", x)[0]))
    return files[-1]


def getSolverStateFile(d):
    if not os.path.exists(d):
        return None
    files = glob.glob(d + "*.solverstate")
    if not files:
        files = glob.glob(d + "/net_*.solverstate")
    if len(files) == 0:
        return None
    files.sort(key=lambda x: int(re.findall(r"\d+", x)[0]))
    return files[-1]

def detectFreeGPU(failOnNone = True):
    if platform.node().split(".")[0] in ("pcgpu1", "pcgpu2", "pcgpu3", "pcgpu4", "pcgpu5"):
        return 0
    freeGPU = subprocess.check_output('nvidia-smi -q | grep "Minor\|Processes" | grep "None" -B1 | tr -d " " | cut -d ":" -f2 | sed -n "1p"',shell=True)
    if freeGPU == "":
        if failOnNone:
            log("!!!!! ERROR: NO FREE GPU FOUND !!!!!")
            sys.exit(1)
        else:
            return None
    else:
        return int(freeGPU)


"""
CV2
"""
def cvImgRead(p):
    img = cv2.imread(p)
    assert img is not None
    return img

def draw3DBBToImage(img, boundingBox, offset = np.array([0,0]), outterOnly = False, everything = False, color = True, linewidth=1):
    assert not (outterOnly and everything)
    boundingBox = map(lambda r: tuple(np.round(r-offset).astype(np.int)), np.array(boundingBox))
    if color:
        color1 = (0xFF, 0xA8, 0x4E)
        color2 = (0x00, 0x0D, 0xFF)
        color3 = (0x00, 0xDA, 0xFF)
    else:
        color1 = (0, 0xFB, 0x87)
        color2 = color1
        color3 = color1
    if everything:
        cv2.line(img, boundingBox[4], boundingBox[7], color1, linewidth, cv2.cv.CV_AA)
        cv2.line(img, boundingBox[6], boundingBox[7], color2, linewidth, cv2.cv.CV_AA)
        cv2.line(img, boundingBox[3], boundingBox[7], color3, linewidth, cv2.cv.CV_AA)
    if not outterOnly:
        cv2.line(img, boundingBox[0], boundingBox[1], color2, linewidth, cv2.cv.CV_AA)
        cv2.line(img, boundingBox[1], boundingBox[2], color1, linewidth, cv2.cv.CV_AA)
        cv2.line(img, boundingBox[1], boundingBox[5], color3, linewidth, cv2.cv.CV_AA)
    cv2.line(img, boundingBox[2], boundingBox[3], color2, linewidth, cv2.cv.CV_AA)
    cv2.line(img, boundingBox[3], boundingBox[0], color1, linewidth, cv2.cv.CV_AA)
    cv2.line(img, boundingBox[0], boundingBox[4], color3, linewidth, cv2.cv.CV_AA)
    cv2.line(img, boundingBox[2], boundingBox[6], color3, linewidth, cv2.cv.CV_AA)
    cv2.line(img, boundingBox[4], boundingBox[5], color2, linewidth, cv2.cv.CV_AA)
    cv2.line(img, boundingBox[5], boundingBox[6], color1, linewidth, cv2.cv.CV_AA)


def collateImages(imgs, targetWidth = 256, cols = 6):
    rowsImages = []
    currentRow = []
    for img in imgs:
        currentRow.append(img)
        if len(currentRow) == cols:
            rowsImages.append(currentRow)
            currentRow = []
    if len(currentRow) > 0:
        rowsImages.append(currentRow)
    rows = []
    maxHeight = 0
    for rowImages in rowsImages:
        transfomed = []
        for img in rowImages:
            scaleFactor = targetWidth/img.shape[1]
            imgResized = cv2.resize(img, dsize=(0,0), fx=scaleFactor, fy=scaleFactor)
            maxHeight = max(maxHeight, imgResized.shape[0])
            transfomed.append(imgResized)
        rowImages = transfomed
        transfomed = []
        for img in rowImages:
            if img.shape[0] < maxHeight:
                img = np.concatenate((np.ones((maxHeight-img.shape[0], targetWidth, 3), dtype=img.dtype)*255, img), axis=0)
            transfomed.append(img)
        rows.append(np.concatenate(transfomed, axis=1))
    if rows[-1].shape[1] != cols*targetWidth:
        rows[-1] = np.concatenate((rows[-1], np.ones((rows[-1].shape[0], targetWidth*cols-rows[-1].shape[1], 3), dtype=rows[-1].dtype)*255), axis=1)
    finalImg = np.concatenate(rows, axis=0)
    return finalImg




def drawLine(img, l, color = (0,255,0), thickness = 2, lineType = cv2.CV_AA):
    pt1 = (0, int((-l[2]/l[1])))
    pt2 = (img.shape[1], int((-l[0]*img.shape[1]-l[2])/l[1]))
    cv2.line(img, pt1, pt2, color = color, thickness = thickness, lineType = lineType)
