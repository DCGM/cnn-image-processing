'''
Created on May 27, 2016

@author: isvoboda
'''

from factory import ObjectFactory
from provider import DataProvider
from processing import DataProcessing
from trainer import Trainer

class Creator(object):
    """
    Create the readers and filters according the configuration file.
    """
    def __init__(self, config):
        self.config = config
        self.f_create = ObjectFactory.create_object
    
    def parse_filters(self, l_filters):
        """
        Parse list of filters.
        """
        new_filters = []
        for fil in l_filters:
            (fil_id, fil_params), = fil.items()
            if fil_params != None:
                new_filters.append(self.f_create(fil_id, **fil_params))
            else:
                new_filters.append(self.f_create(fil_id))
        
        return new_filters 
    
    def parse_tuples(self, tuples):
        """
        Parse list of filter tuples.
        """
        tup_filters = []
        filters = None
        for tup_f in tuples:
            parameters = dict()
            for (key_id, val) in tup_f.items():
                for (param_id, param_val) in val.items():
                    if param_id == "Filters" or param_id == "Readers":
                        filters = self.parse_filters(param_val)
                    elif param_id == "Parameters" and param_val is not None:
                        parameters.update(param_val)
                # Fixme: if parameters == None, than fails
                parameters['filters'] = filters
                assert parameters != None
                t_fils = self.f_create(key_id, **parameters)
                tup_filters.append(t_fils)

        return tup_filters

    def create_provider(self):
        """
        Creates provider.
        """
        tuple_filters = self.parse_tuples(self.config['Provider']['TFilters'])
        parameters = self.config['Provider']['Parameters']
        return DataProvider(t_readers=tuple_filters, **parameters)
        
    def create_processing(self):
        """
        Creates processing
        """
        tuple_filters = self.parse_tuples(self.config['Processing']
                                          ['TFilters'])
        parameters = self.config['Processing']['Parameters']
        return DataProcessing(t_filters=tuple_filters, **parameters)
    
    def create_training(self):
        """
        Creates the trainer.
        """
        return Trainer(**self.config['Trainer'])