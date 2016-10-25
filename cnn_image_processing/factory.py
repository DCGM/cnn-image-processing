from __future__ import print_function


class FilterFactory(object):

    factories = {}

    @classmethod
    def register(cls, name, impl):
        if name in cls.factories:
            raise RuntimeError('A filter was registered multiple times.')
        cls.factories[name] = impl

    @classmethod
    def print_registered_filters(cls):
        print('Registered filters.')
        for filterName in cls.factories:
            print(filterName)


    '''Pass': filters.Pass,
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
                 'ClipValues': filters.ClipValues
                 }
        '''

    @classmethod
    def create_object(cls, id_object, config):
        """
        Creates the object according the id_filter.
        In case the id_object has not yet been registered than add it.
        """
        if id_object not in cls.factories:
            raise AttributeError('Unknown filter in factory: "{}"'.format(id_object))
        return cls.factories[id_object](config=config)
