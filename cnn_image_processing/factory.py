'''
Created on May 27, 2016

@author: isvoboda
'''

import filters

class ObjectFactory(object):

    """
    Factory class to create several object.
    """
    factories = { 'Pass': filters.Pass,
                  'TFilter':  filters.TFilter,
                  'TCropCoef8ImgFilter': filters.TCropCoef8ImgFilter,
                  'TReader': filters.TFilter,
                  'CoefNpyTxtReader': filters.CoefNpyTxtReader,
                  'ImageReader': filters.ImageReader,
                  'ImageX8Reader': filters.ImageX8Reader,
                  'Crop': filters.Crop,
                  'LTCrop': filters.LTCrop,
                  'Label': filters.Label,
                  'Mul': filters.Mul,
                  'Div': filters.Div,
                  'Add': filters.Add,
                  'Sub': filters.Sub,
                  'JPGBlockReshape': filters.JPGBlockReshape,
                  'MullQuantTable': filters.MullQuantTable,
                  'Preview': filters.Preview,
                  'DecodeDCT': filters.DecodeDCT }  # Static attribute

    @classmethod
    def create_object(cls, id_object, **kwargs):
        """
        Creates the object according the id_filter.
        In case the id_object has not yet been registered than add it.
        """
        if not cls.factories.has_key(id_object):
            # ToDo eval is not safe - anything could be inserted in.
            cls.factories[id_object] = \
                                           cls.get_class(id_object)
            print(id_object)
        return cls.factories[id_object](**kwargs)

    @staticmethod
    def get_class(class_name):
        """
        Returns the class of the name class_name

        Args:
          class_name: name of the class to be referenced (eg module.class)
        Return:
          The reference to the class of name class_name
        """
        parts = class_name.split(".")
        module = ".".join(parts[:-1])
        print(module)
        mod = __import__(module)
        for component in parts[1:]:
            mod = getattr(mod, component)
        return mod
