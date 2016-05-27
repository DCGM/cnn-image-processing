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
import cnn_image_processing as cip

LOGGER = logging.getLogger("cnn_image_processing")

def main(argv):
    '''
    Entry point
    Args:
        argv: list of command line arguments.
    '''
    parser = argparse.ArgumentParser(description="Train the coef CNN.")
    parser.add_argument("-c", "--conf-file", help="Configuration file",
                        type=str, required=True )
    parser.add_argument("-s", "--solver-file", help="Solver file", type=str,
                         required=True)
    parser.add_argument("-f", "--file-list", help="File list", type=str,
                         required=True)
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
    solver_file = args.solver_file
    data_file = args.file_list

    # Open, parse and print the configuration file
    with open(config_file) as cf_file:
        config = yaml.safe_load(cf_file)
        print (yaml.dump(config))
    
    # Create communication queues
    readers_queue = multiprocessing.Queue(64)
    data_queue = multiprocessing.Queue(1024)
    
    # Create and initialize main objects
    creator = cip.Creator(config=config)
    
    d_provider = creator.create_provider()
    d_provider.file_list = data_file
    d_provider.out_queue = readers_queue
    
    d_processing = creator.create_processing()
    d_processing.in_queue = readers_queue
    d_processing.out_queue = data_queue

    proc_trainer = creator.create_training()
    proc_trainer.in_queue=data_queue
    proc_trainer.solver_file = solver_file

    # Run the whole magic
    d_provider.start()
    d_processing.start()
    proc_trainer.start()
    
    proc_trainer.join()
    
if __name__ == "__main__":
    main(sys.argv)
