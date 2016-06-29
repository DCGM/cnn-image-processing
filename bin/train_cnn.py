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
from collections import OrderedDict
import cnn_image_processing as ci
import signal
signal.signal(signal.SIGINT, lambda x,y: sys.exit(1))

LOGGER = logging.getLogger("cnn_image_processing")


def parse_phase(conf):
    dmodules = {}
    
    creator = ci.Creator
    
    pque_size = 5
    if 'provider_queue_size' in conf:
        pque_size = conf['provider_queue_size'] 
    
    sque_size = 512
    if 'sample_queue_size' in conf:
        sque_size = conf['sample_queue_size']     
         
    dmodules['pque'] = multiprocessing.Queue(pque_size)
    dmodules['sque'] = multiprocessing.Queue(sque_size)
    
    if 'Provider' in conf:
        dmodules['provider'] = creator.create_provider(conf['Provider'])
        dmodules['provider'].out_queue = dmodules['pque']
    else:
        dmodules['provider'] = None
#     train_provider.file_list = train_list
    
    if 'Sampler' in conf:
        dmodules['sampler'] = creator.create_sampler(conf['Sampler'])
        dmodules['sampler'].in_queue = dmodules['pque']
        dmodules['sampler'].out_queue = dmodules['sque']
    else:
        dmodules['sampler'] = None
    
    return dmodules

def parse_config(conf=None):
    creator = ci.Creator
    app = {}
    
    app['Train'] = parse_phase(conf['Train'])
    app['Train']['provider'].out_queue = app['Train']['pque']
    app['Train']['sampler'].in_queue = app['Train']['pque']
    app['Train']['sampler'].out_queue = app['Train']['sque']
    in_ques = []
    
    if 'Test' in conf:
        test_nets = OrderedDict()
        test_net_list = [test_net.keys()[0] for test_net in conf['Test']]
        test_net_list.sort()
        for i_key, net_key in enumerate(test_net_list):       
            test_nets[net_key] = parse_phase(conf['Test'][i_key][net_key])
            
            if test_nets[net_key]['provider'] == None:
                tprovider = creator.create_provider(conf['Train']['Provider'])
                tprovider.out_queue = test_nets[net_key]['pque']
                test_nets[net_key]['provider'] = tprovider
            if test_nets[net_key]['sampler'] == None:
                tsampler = creator.create_sampler(['Train']['Sampler'])
                tsampler.in_queue = test_nets[net_key]['pque']
                tsampler.out_queue = test_nets[net_key]['sque']
                test_nets[net_key]['sampler'] = tsampler
                    
            in_ques.append(test_nets[net_key]['sque'])
        app['Test'] = test_nets
    
    app['Trainer'] = creator.create_trainer(conf['Trainer'])
    app['Trainer'].train_in_queue = app['Train']['sque']
    app['Trainer'].test_in_queue = in_ques
    
    return app

def main(argv):
    '''
    Entry point
    Args:
        argv: list of command line arguments.
    '''
    parser = argparse.ArgumentParser(description="Train the cnn")
    parser.add_argument("-c", "--conf-file", help="Configuration file",
                        type=str, required=True )
    parser.add_argument("-s", "--solver-file", help="Solver file", type=str,
                         required=True)
    parser.add_argument("-tr", "--train-list", help="Training file list",
                        type=str, required=True)
    parser.add_argument("-te", "--test-lists", help="Testing file lists",
                        nargs='*', type=str, required=False, default=None)
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
    test_lists = args.test_lists

    # Open, parse and print the configuration file
    with open(config_file) as cf_file:
        conf = yaml.safe_load(cf_file)
        print (yaml.dump(conf))
    
    app = parse_config(conf)
    
    app['Train']['provider'].file_list = train_list
    app['Train']['provider'].start()
    app['Train']['sampler'].start()
    if test_lists is not None:
        assert(len(test_lists) == len(app['Test']))
        for i_test, test_k in enumerate(app['Test']):
            app['Test'][test_k]['provider'].file_list = test_lists[i_test]
            app['Test'][test_k]['provider'].start()
            app['Test'][test_k]['sampler'].start()
    
    app['Trainer'].solver_file = solver_file
    app['Trainer'].start()
    app['Trainer'].join()
    
if __name__ == "__main__":
    main(sys.argv)
