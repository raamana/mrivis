
import numpy as np
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
import matplotlib
import matplotlib.pyplot as plt

def get_freesurfer_cortical_LUT():
    """
    Subset of Freesurfer ColorLUT for cortical labels

    Original at
    https://surfer.nmr.mgh.harvard.edu/fswiki/FsTutorial/AnatomicalROI/FreeSurferColorLUT
    """

    LUT = [ [0, 0, 0],
            [25, 5, 25],
            [25,100,40],
            [125,100,160],
            [100,25,0],
            [120,70,50],
            [220,20,100],
            [220,20,10],
            [180,220,140],
            [220,60,220],
            [180,40,120],
            [140,20,140],
            [20,30,140],
            [35,75,50],
            [225,140,140],
            [200,35,75],
            [160,100,50],
            [20,220,60],
            [60,220,60],
            [220,180,140],
            [20,100,50],
            [220,60,20],
            [120,100,60],
            [220,20,20],
            [220,180,220],
            [60,20,220],
            [160,140,180],
            [80,20,140],
            [75,50,125],
            [20,220,160],
            [20,180,140],
            [140,220,220],
            [80,160,20],
            [100,0,100],
            [70,70,70],
            [150,150,200],
            [220,216,20]
           ]

    LUT = np.array(LUT)

    # to make them range from 0 to 1, required by python (rgb values must be between 0 and 1)
    LUT = LUT/255

    return LUT


def get_freesurfer_subcortical_LUT():
    """
    Subset of Freesurfer ColorLUT for cortical labels

    Original at
    https://surfer.nmr.mgh.harvard.edu/fswiki/FsTutorial/AnatomicalROI/FreeSurferColorLUT
    """

    return plt.cm.Set1 # tab20_r


def get_freesurfer_cmap(vis_type):
    """Provides different colormaps for different visualization types."""

    if vis_type in ('cortical_volumetric'):
            LUT = get_freesurfer_cortical_LUT()
            fs_cmap = ListedColormap(LUT)
    elif vis_type in ('labels', 'label_set'):
        binary = np.array([0, 0, 0, 1])
        fs_cmap = plt.cm.Dark2(np.linspace(0, 1, 40)) # other options Set1, tab10
        colors = np.vstack((binary, fs_cmap))
        fs_cmap = ListedColormap(colors, 'my_colormap', N=256)
    else:
        raise NotImplementedError('color map for the visualization type {} has not been implemented!'.format(vis_type))

    return fs_cmap


