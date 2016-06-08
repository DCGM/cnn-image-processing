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

PROVIDER_QUEUE_SIZE = 20
SAMPLE_QUEUE_SIZE = 1024


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
    parser.add_argument("-tr", "--train-list", help="Training file list",
                        type=str, required=True)
    parser.add_argument("-te", "--test-list", help="Testing file list",
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
    
    creator = cip.Creator()
    
    trainer = creator.create_trainer(config['Trainer'])
    trainer.solver_file = solver_file
    
    if 'Train' in config:
        train_provider_queue = multiprocessing.Queue(PROVIDER_QUEUE_SIZE)
        
        train_provider = creator.create_provider(config['Train']['Provider'])
        train_provider.file_list = train_list
        train_provider.out_queue = train_provider_queue
        train_provider.start()
    
        train_samples_queue = multiprocessing.Queue(SAMPLE_QUEUE_SIZE)
    
        train_sampler = creator.create_sampler(config['Train']['Sampler'])
        train_sampler.in_queue = train_provider_queue
        train_sampler.out_queue = train_samples_queue
        train_sampler.start()
        
        trainer.train_in_queue = train_samples_queue
    
    if 'Test' in config and test_list is not None:
        test_provider_queue = multiprocessing.Queue(PROVIDER_QUEUE_SIZE)
        
        test_provider = creator.create_provider(config['Test']['Provider'])
        test_provider.file_list = test_list
        test_provider.out_queue = test_provider_queue
        test_provider.start()
        
        test_samples_queue = multiprocessing.Queue(SAMPLE_QUEUE_SIZE)
    
        test_sampler = creator.create_sampler(config['Test']['Sampler'])
        test_sampler.in_queue = test_provider_queue
        test_sampler.out_queue = test_samples_queue
        test_sampler.start()
        
        trainer.test_in_queue = test_samples_queue
    
    # Run the trainer
    
    trainer.start()
    trainer.join()
    
if __name__ == "__main__":
    main(sys.argv)
