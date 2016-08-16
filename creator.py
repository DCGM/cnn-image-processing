'''
Created on May 27, 2016

@author: isvoboda
'''

from __future__ import print_function

from .factory import ObjectFactory
from .provider import Provider
from .sampler import Sampler
from .trainer import Trainer
from .fcn import FCN
from .classifier import Classifier


class Creator(object):
    """
    Create the readers and filters according the configuration file.
    """
    f_create = ObjectFactory.create_object

    @classmethod
    def parse_filters(cls, l_filters):
        """
        Parse list of filters.
        """
        new_filters = []
        for fil in l_filters:
            (fil_id, fil_params), = fil.items()
            if fil_params != None:
                new_filters.append(cls.f_create(fil_id, **fil_params))
            else:
                new_filters.append(cls.f_create(fil_id))

        return new_filters

    @classmethod
    def parse_tuples(cls, tuples):
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
                        filters = cls.parse_filters(param_val)
                    elif param_id == "Parameters" and param_val is not None:
                        parameters.update(param_val)
                # Fixme: if parameters == None, than fails
                parameters['filters'] = filters
                assert parameters != None
                t_fils = cls.f_create(key_id, **parameters)
                tup_filters.append(t_fils)

        return tup_filters

    @classmethod
    def create_provider(cls, config):
        """
        Creates provider.
        """
        tuple_filters = cls.parse_tuples(config['TFilters'])
        parameters = config['Parameters']
        return Provider(t_readers=tuple_filters, **parameters)

    @classmethod
    def create_sampler(cls, config):
        """
        Creates sampler
        """
        if config['TFilters']:
            tuple_filters = cls.parse_tuples(config['TFilters'])
        else:
            tuple_filters = []
        parameters = config['Parameters']
        return Sampler(t_filters=tuple_filters, **parameters)

    @classmethod
    def create_trainer(cls, config):
        """
        Creates the trainer.
        """
        return Trainer(**config)

    @classmethod
    def create_fcn(cls, config):
        """
        Creates the trainer.
        """
        return FCN(**config)

    @classmethod
    def create_class(cls, config):
        """
        Creates the trainer.
        """
        return Classifier(**config)
