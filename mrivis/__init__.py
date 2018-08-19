# -*- coding: utf-8 -*-

"""Top-level package for mrivis."""

__author__ = """Pradeep Reddy Raamana"""
__email__ = 'raamana@gmail.com'

__all__ = ['checkerboard', 'color_mix', 'voxelwise_diff', 'collage', 'aseg_on_mri',
           'Collage', 'SlicePicker', 'color_maps']

from sys import version_info
if version_info.major > 2:
    from mrivis.workflow import checkerboard, color_mix, voxelwise_diff, collage, aseg_on_mri
    from mrivis.base import Collage, SlicePicker, MiddleSlicePicker, MidCollage, Carpet
    from mrivis import color_maps
else:
    # from .mrivis import checkerboard
    raise NotImplementedError('mrivis requires Python 3+.')

from ._version import get_versions
__version__ = get_versions()['version']
del get_versions
