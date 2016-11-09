from __future__ import print_function

import logging
import copy
from factory import FilterFactory
import ruamel.yaml


class ContinuePipeline(Exception):
    '''
    Signalizes that pipeline process should start from the beginning.
    Can be used in case of non-fatal errors (e.g. file missing).
    '''
    pass


class TerminatePipeline(Exception):
    '''
    Signalizes that pipeline process should be terminated.
    Should not be used in case of errors.
    '''
    pass


class parameter(object):
    def __init__(self, field, required=False, default=None, parser=lambda x: x, help=''):
        self.field = field
        self.required = required
        self.default = default
        self.parser = parser

    def __call__(self, config):
        if self.field not in config:
            if self.required:
                message = ' CONFIGURATION PARSING ERROR: Missing required option "{}" at line {}.'.format(self.field, config.lc.line)
                raise AttributeError(message)
            return self.default

        try:
            value = self.parser(config[self.field])
        except Exception as ex:
            message = ' CONFIGURATION PARSING ERROR: Unable to parse option "{}" with value "{}" line {}. '.format(self.field, ruamel.yaml.dump(config[self.field], Dumper=ruamel.yaml.RoundTripDumper), config.lc.line)
            raise AttributeError(message + str(ex))
        return value

    def __str__(self):
        return 'Param: {}:{}, {}:{}, {}:{}'.format(
            'field', self.field,
            'required', self.required,
            'default', self.default)


class FactoryRegister(type):
    ''' This is a metaclass (type) used to register all Configurable classes
    with filter factory
    '''
    def __new__(cls, clsname, bases, attrs):
        newClass = super(FactoryRegister, cls).__new__(cls, clsname, bases, attrs)
        print('REGISTER', clsname)
        FilterFactory.register(clsname, newClass)
        return newClass


class Configurable(object):
    #so that all child classes register automatically with factory
    __metaclass__ = FactoryRegister

    def __init__(self):
        self.params = []
        self.log = logging.getLogger(__name__ + "." + type(self).__name__)

    def init(self):
        pass
    def addParams(self):
        pass

    def parseParams(self, config):
        try:
            configCopy = copy.copy(config)
            for param in self.params:
                value = param(config)
                setattr(self, param.field, value)
                if param.field in configCopy:
                    del configCopy[param.field]
        except AttributeError as ex:
            self.log.error(ex)
            self.printParams()
            raise ex

        if configCopy:
            msg = 'Some params were not parsed. Remaining "{}"'.format(str(configCopy))
            self.log.error(msg)
            self.printParams()
            raise AttributeError(msg)

    def printParams(self):
        self.log.info('Accepted params are:')
        for param in self.params:
            self.log.info(str(param))
