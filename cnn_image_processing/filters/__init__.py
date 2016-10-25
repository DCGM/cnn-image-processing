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

#FilterFactory.print_registered_filters()
