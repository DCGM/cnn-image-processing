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
    parser.add_argument("-l", "--train-list", help="Training file list",
                        type=str, required=True)
    parser.add_argument("-tl", "--test-list", help="Testing file list",
                        type=str, required=False, default=None)
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
    train_list = args.train_list
    test_list = args.test_list

    # Open, parse and print the configuration file
    with open(config_file) as cf_file:
        config = yaml.safe_load(cf_file)
        print (yaml.dump(config))
    
    # Create communication queues
    train_readers_queue = multiprocessing.Queue(64)
    train_data_queue = multiprocessing.Queue(1024)
 
    
    # Create and initialize main objects
    creator = cip.Creator(config=config)
    
    d_provider = creator.create_provider()
    d_provider.file_list = train_list
    d_provider.out_queue = train_readers_queue
    
    d_processing = creator.create_processing()
    d_processing.in_queue = train_readers_queue
    d_processing.out_queue = train_data_queue
  
    proc_trainer = creator.create_training()
    proc_trainer.in_queue=train_data_queue
    proc_trainer.solver_file = solver_file

    if args.test_list != None:
        test_readers_queue = multiprocessing.Queue(64)
        test_data_queue = multiprocessing.Queue(1024)
        
        test_provider = creator.create_provider()
        test_provider.file_list = test_list
        test_provider.out_queue = test_readers_queue 
        
        test_processing = creator.create_processing()
        test_processing.in_queue = test_readers_queue
        test_processing.out_queue = test_data_queue
        proc_trainer.test_in_queue = test_data_queue
        
        test_provider.start()
        test_processing.start()


    # Run the whole magic
    d_provider.start()
    d_processing.start()
    proc_trainer.start()
    
    proc_trainer.join()
    
if __name__ == "__main__":
    main(sys.argv)
