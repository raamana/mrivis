# -*- coding: utf-8 -*-

"""Top-level package for mrivis."""

__author__ = """Pradeep Reddy Raamana"""
__email__ = 'raamana@gmail.com'

__all__ = ['checkerboard', 'color_mix', 'voxelwise_diff', 'collage',
           'Collage', 'SlicePicker', 'Carpet', 'MiddleSlicePicker', 'MidCollage',
           'aseg_on_mri', 'color_maps']

from sys import version_info

if version_info.major > 2:
    from mrivis.workflow import (checkerboard, color_mix, voxelwise_diff, collage,
                                 aseg_on_mri)
    from mrivis.base import (Collage, MidCollage,
                             SlicePicker, MiddleSlicePicker,
                             Carpet)
    from mrivis import color_maps
else:
    # from .mrivis import checkerboard
    raise NotImplementedError('mrivis requires Python 3 or higher. Please upgrade.')


del version_info

try:
    from ._version import __version__
except ImportError:
    __version__ = "0+unknown"
