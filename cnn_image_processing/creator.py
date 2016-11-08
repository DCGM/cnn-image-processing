from __future__ import print_function

from .factory import FilterFactory


class Creator(object):
    """
    Create objects according the configuration file.
    """

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
            stage = cls.parse_filters(stage_config)
            pipeline.append(stage)
        return pipeline

    @classmethod
    def parse_config(cls, config):
        processes = []
        FilterFactory.print_registered_filters()
        for c in config:
            name = c.keys()[0]
            configuration = c.values()[0]
            processes.append(FilterFactory.create_object(name, configuration))
        return processes




