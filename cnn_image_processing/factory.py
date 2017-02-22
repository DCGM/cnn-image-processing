from __future__ import print_function


class FilterFactory(object):

    factories = {}

    @classmethod
    def register(cls, name, impl):
        if name in cls.factories:
            raise RuntimeError('A filter "{}" was registered multiple times.'.format(name))
        cls.factories[name] = impl

    @classmethod
    def print_registered_filters(cls):
        '''
        This method is automatically called for all classes inheriting
        from class Configurable due to FactoryRegister metaclass.
        '''
        print('Registered filters.')
        for filterName in cls.factories:
            print(filterName)

    @classmethod
    def create_object(cls, id_object, config):
        """
        Creates the object according the id_filter.
        In case the id_object has not yet been registered than add it.
        """
        if id_object not in cls.factories:
            if 'lc' in config:
                raise AttributeError(' Unknown filter in configuration: "{}" at line {}'.format(id_object, config.lc.line))
            else:
                raise AttributeError(' Unknown filter in configuration: "{}"'.format(id_object))

        return cls.factories[id_object](config=config)
