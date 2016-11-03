'''
Created on May 27, 2016

@author: isvoboda
'''

from __future__ import print_function

from .process import Process
from .factory import FilterFactory
from .sampler import Sampler
from .trainer import Trainer
from .fcn import FCN


class Creator(object):
    """
    Create objects according the configuration file.
    """
    create_filter = FilterFactory.create_object

    @classmethod
    def parse_filters(cls, l_filters):
        """
        Parse list of filters.
        """
        new_filters = []
        for fil in l_filters:
            (fil_id, fil_params), = fil.items()
            if fil_params is None:
                fil_params = {}
            new_filter = FilterFactory.create_object(fil_id, fil_params)
            new_filters.append(new_filter)
        return new_filters

    @classmethod
    def parse_pipeline(cls, pipeline_config):
        """
        Parse list of filter tuples.
        """
        pipeline = []
        for stage_config in pipeline_config:
            pipeline.append(cls.parse_filters(stage_config))
        return pipeline

    @classmethod
    def create_provider(cls, config):
        """
        Creates provider.
        """
        pipeline_config = cls.parse_pipeline(config['pipeline'])
        return Process(pipeline=pipeline_config, name=config['name'])

    @classmethod
    def create_sampler(cls, config):
        """
        Creates sampler
        """
        tuple_filters = cls.parse_tuples(config['TFilters'])
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

