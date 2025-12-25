"""Top-level package for mrivis."""

__author__ = """Pradeep Reddy Raamana"""
__email__ = 'raamana@gmail.com'

__all__ = ['checkerboard', 'color_mix', 'voxelwise_diff',
           'Collage', 'SlicePicker', 'Carpet', 'MiddleSlicePicker', 'MidCollage',
           'aseg_on_mri', 'color_maps']

from mrivis.workflow import (checkerboard, color_mix, voxelwise_diff,
                             aseg_on_mri)
from mrivis.base import (Collage, MidCollage,
                         SlicePicker, MiddleSlicePicker,
                         Carpet)
from mrivis import color_maps

try:
    from ._version import __version__
except ImportError:
    __version__ = "0+unknown"
