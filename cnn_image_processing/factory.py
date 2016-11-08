'''
Created on May 27, 2016

@author: isvoboda
'''

from __future__ import print_function

from . import filters


class ObjectFactory(object):

    """
    Factory class to create several object.
    """
    factories = {'Pass': filters.Pass,
                 'TFilter':  filters.TFilter,
                 'THorizontalFilter': filters.THorizontalFilter,
                 'TCropCoef8ImgFilter': filters.TCropCoef8ImgFilter,
                 'TReader': filters.TFilter,
                 'CoefNpyTxtReader': filters.CoefNpyTxtReader,
                 'ImageReader': filters.ImageReader,
                 'ImageX8Reader': filters.ImageX8Reader,
                 'TupleReader': filters.TupleReader,
                 'Crop': filters.Crop,
                 'LTCrop': filters.LTCrop,
                 'Label': filters.Label,
                 'Mul': filters.Mul,
                 'Div': filters.Div,
                 'Add': filters.Add,
                 'Sub': filters.Sub,
                 'JPGBlockReshape': filters.JPGBlockReshape,
                 'MulQuantTable': filters.MulQuantTable,
                 'Preview': filters.Preview,
                 'DecodeDCT': filters.DecodeDCT,
                 'CodeDCT': filters.CodeDCT,
                 'Pad8': filters.Pad8,
                 'PadCoefMirror': filters.PadCoefMirror,
                 'JPEG': filters.JPEG,
                 'Resize': filters.Resize,
                 'Round': filters.Round,
                 'FixedCrop': filters.FixedCrop,
                 'ShiftImg': filters.ShiftImg,
                 'GammaCorrection': filters.GammaCorrection,
                 'ReduceContrast': filters.ReduceContrast,
                 'ColorBalance': filters.ColorBalance,
                 'Noise': filters.Noise,
                 'VirtualCamera': filters.VirtualCamera,
                 'Flip': filters.Flip,
                 'Copy': filters.Copy,
                 'ClipValues': filters.ClipValues,
                 'CameraReader': filters.CameraReader
                 }

    @classmethod
    def create_object(cls, id_object, **kwargs):
        """
        Creates the object according the id_filter.
        In case the id_object has not yet been registered than add it.
        """
        if not cls.factories.has_key(id_object):
            print("Not known filter: {}".format(id_object))
            return None
            # ToDo eval is not safe - anything could be inserted in.
#             cls.factories[id_object] = \
#                 cls.get_class(id_object)
#             print(id_object)
        return cls.factories[id_object](**kwargs)

#     @staticmethod
#     def get_class(class_name):
#         """
#         Returns the class of the name class_name
#
#         Args:
#           class_name: name of the class to be referenced (eg module.class)
#         Return:
#           The reference to the class of name class_name
#         """
#         parts = class_name.split(".")
#         module = ".".join(parts[:-1])
#         print(module)
#         mod = __import__(module)
#         for component in parts[1:]:
#             mod = getattr(mod, component)
#         return mod
