"""
Filters, readers, and writers
"""

from .. import FilterFactory

import readers
import filters

FilterFactory.register('CoefNpyTxtReader', readers.CoefNpyTxtReader)
FilterFactory.register('ImageX8Reader', readers.ImageX8Reader)
FilterFactory.register('TupleReader', readers.TupleReader)
FilterFactory.register('ImageReader', readers.ImageReader)
FilterFactory.register('ListFileReader', readers.ListFileReader)

FilterFactory.register('Preview', filters.Preview)
FilterFactory.register('HorizontalPassPackets', filters.HorizontalPassPackets)

FilterFactory.register('Label', filters.Label)
FilterFactory.register('MulAdd', filters.MulAdd)
FilterFactory.register('Pass', filters.Pass)
FilterFactory.register('JPEG', filters.JPEG)
FilterFactory.register('Resize', filters.Resize)
FilterFactory.register('CustomFunction', filters.CustomFunction)
FilterFactory.register('CentralCrop', filters.CentralCrop)
FilterFactory.register('Round', filters.Round)


#FilterFactory.print_registered_filters()
