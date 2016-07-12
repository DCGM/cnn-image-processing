#!/usr/bin/env python
'''
Created on Jun 6, 2016

@author: isvoboda
'''

from __future__ import print_function

import cv2
import argparse
import logging
import os
import sys

LOGGER = logging.getLogger(__name__)


def main(argv):
    '''
    Entry point
    Args:
        argv: list of command line arguments.
    '''
    parser = argparse.ArgumentParser(description="Store files to jpeg.")
    parser.add_argument("-f", "--file-list", help="File list", type=str,
                        required=True)
    parser.add_argument("-q", "--quality", help="Set the jpeg quality.",
                        type=int, default=90)
    parser.add_argument( "-d", "--dir-path",
                         help="Set the destination directory path.", type=str)

    args = parser.parse_args()

    LOGGER.setLevel(logging.INFO)
    logging.basicConfig()

    # Print the arguments
    for key, val in vars(args).iteritems():
        print("{}: {}".format(key, val))

    file_list = args.file_list
    quality = args.quality
    dir_path = args.dir_path

    with open(file_list) as flist:
        for line in flist:
            LOGGER.info("Reading {}".format(line))
            filename, _ = os.path.splitext(line)
            filename = os.path.basename(filename) + ".jpg"
            img = cv2.imread(line.strip())
            w_line = os.path.join(dir_path, filename)
            LOGGER.info("Writing {}".format(w_line))
#             cv2.imshow("a", img)
#             cv2.waitKey()
            cv2.imwrite(w_line, img, [cv2.IMWRITE_JPEG_QUALITY, quality])

if __name__ == "__main__":
    main(sys.argv)

