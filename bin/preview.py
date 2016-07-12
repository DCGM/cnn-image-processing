#!/usr/bin/env python
'''
Created on Apr 7, 2016

@author: isvoboda
'''
import sys
import multiprocessing
import logging
import yaml
import argparse
import cnn_image_processing as ci

LOGGER = logging.getLogger("cnn_image_processing")


def main(argv):
    '''
    Entry point
    Args:
        argv: list of command line arguments.
    '''
    parser = argparse.ArgumentParser(description="Train the coef CNN.")
    parser.add_argument("-c", "--conf-file", help="Configuration file",
                        type=str, required=True)
    parser.add_argument("-l", "--list", help="Training file list",
                        type=str, required=True)
    parser.add_argument("-v", "--verbose", help="Set the verbose mode.",
                        action="store_true")

    args = parser.parse_args()

    # Print the arguments
    for key, val in vars(args).iteritems():
        print("{}: {}".format(key, val))

    # Initialize logging
    if args.verbose:
        LOGGER.setLevel(logging.DEBUG)
    else:
        LOGGER.setLevel(logging.INFO)

    logging.basicConfig()

    config_file = args.conf_file
    file_list = args.list

    # Open, parse and print the configuration file
    with open(config_file) as cf_file:
        config = yaml.safe_load(cf_file)
        print (yaml.dump(config))

    creator = ci.Creator()

    train_provider = creator.create_provider(config['Provider'])
    train_provider.file_list = file_list
    train_provider.run()

if __name__ == "__main__":
    main(sys.argv)
