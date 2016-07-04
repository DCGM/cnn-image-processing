#!/usr/bin/env python

'''
Created on Jun 28, 2016

@author: isvoboda
'''
import sys
import multiprocessing
import logging
import yaml
import argparse
import cnn_image_processing as ci
import signal
signal.signal(signal.SIGINT, lambda x,y: sys.exit(1))

LOGGER = logging.getLogger("cnn_image_processing")

def parse_config(conf=None):
    creator = ci.Creator
    app = {}
    
    pque_size = 5
    if 'provider_queue_size' in conf:
        pque_size = conf['provider_queue_size'] 
           
    app['provider'] = creator.create_provider(conf['Provider'])
    app['provider'].out_queue = multiprocessing.Queue(pque_size)
    
    app['fcn'] = creator.create_fcn(conf['FCN'])
    app['fcn'].in_queue = app['provider'].out_queue
    return app

def main(argv):
    '''
    Entry point
    Args:
        argv: list of command line arguments.
    '''
    parser = argparse.ArgumentParser(description="Train the cnn")
    parser.add_argument("-c", "--conf-file", action='store', type=str,
                        choices=None, required=True, help="Configuration file",
                        metavar=None, dest='conf_file' )
    
    parser.add_argument("-d", "--deploy-file", action='store', type=str,
                        choices=None, required=True, help="Solver file",
                        metavar=None, dest='deploy_file')
    
    parser.add_argument("-cw", "--caffe-weights", action='store', type=str,
                        choices=None, required=True, help="Caffe weights file",
                        metavar=None, dest='caffe_weights')
    
    parser.add_argument("-v", "--verbose", action="store_true", required=False,
                        help="Set the verbose mode.", dest='verbose')
    
    parser.add_argument("-l", "--file-list", action='store', type=str,
                        help="File list", required=True,
                        dest='file_list')
      
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
    deploy_file = args.deploy_file
    file_list = args.file_list
    caffe_weights = args.caffe_weights

    # Open, parse and print the configuration file
    with open(config_file) as cf_file:
        conf = yaml.safe_load(cf_file)
        print (yaml.dump(conf))
    
    app = parse_config(conf)
    
    app['provider'].file_list = file_list
    app['provider'].start()
    
    app['fcn'].deploy = deploy_file
    app['fcn'].caffe_weights = caffe_weights
    app['fcn'].start()
    app['fcn'].join()
    
if __name__ == "__main__":
    main(sys.argv)
