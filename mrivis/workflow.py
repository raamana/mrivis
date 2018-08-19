"""
mrivis: Tools to comapre the similarity of two 3d images (structural, functional or parametric maps)

Options include checker board, red green mixer and voxel-wise difference maps.

"""
from functools import partial

from mrivis.base import Collage
from mrivis.color_maps import get_freesurfer_cmap
from mrivis.utils import _diff_image, check_params, check_patch_size, crop_image, \
    crop_to_extents, crop_to_seg_extents, get_axis, pick_slices, read_image, scale_0to1, \
    scale_images_0to1

__all__ = ['checkerboard', 'color_mix', 'voxelwise_diff', 'collage']

import numpy as np
from matplotlib import pyplot as plt, colors, cm
import matplotlib as mpl


def checkerboard(img_spec1=None,
                 img_spec2=None,
                 patch_size=10,
                 view_set=(0, 1, 2),
                 num_slices=(10,),
                 num_rows=2,
                 rescale_method='global',
                 background_threshold=0.05,
                 annot=None,
                 padding=5,
                 output_path=None,
                 figsize=None, ):
    """
    Checkerboard mixer.

    Parameters
    ----------
    img_spec1 : str or nibabel image-like object
        MR image (or path to one) to be visualized

    img_spec2 : str or nibabel image-like object
        MR image (or path to one) to be visualized

    patch_size : int or list or (int, int) or None
        size of checker patch (either square or rectangular)
        If None, number of voxels/patch are chosen such that,
            there will be 7 patches through the width/height.

    view_set : iterable
        Integers specifying the dimensions to be visualized.
        Choices: one or more of (0, 1, 2) for a 3D image

    num_slices : int or iterable of size as view_set
        number of slices to be selected for each view
        Must be of the same length as view_set,
            each element specifying the number of slices for each dimension.
            If only one number is given, same number will be chosen for all dimensions.

    num_rows : int
        number of rows (top to bottom) per each of 3 dimensions

    rescale_method : bool or str or list or None
        Range to rescale the intensity values to
        Default: 'global', min and max values computed based on ranges from both images.
        If false or None, no rescaling is done (does not work yet).

    background_threshold : float or str
        A threshold value below which all the background voxels will be set to zero.
        Default : 0.05. Other option is a string specifying a percentile: '5%', '10%'.
        Specify None if you don't want any thresholding.

    annot : str
        Text to display to annotate the visualization

    padding : int
        number of voxels to pad around each panel.

    output_path : str
        path to save the generate collage to.

    figsize : list
        Size of figure in inches to be passed on to plt.figure() e.g. [12, 12] or [20, 20]

    Returns
    -------
    fig : figure handle
        handle to the collage figure generated.

    """

    img_one, img_two = _preprocess_images(img_spec1,
                                          img_spec2,
                                          rescale_method=rescale_method,
                                          bkground_thresh=background_threshold,
                                          padding=padding)

    display_params = dict(interpolation='none', aspect='auto', origin='lower',
                          cmap='gray', vmin=0.0, vmax=1.0)

    mixer = partial(_checker_mixer, checker_size=patch_size)
    collage = Collage(view_set=view_set, num_slices=num_slices, num_rows=num_rows,
                      figsize=figsize, display_params=display_params)
    collage.transform_and_attach((img_one, img_two), func=mixer)
    collage.save(output_path=output_path, annot=annot)

    return collage


def color_mix(img_spec1=None,
              img_spec2=None,
              alpha_channels=None,
              color_space='rgb',
              view_set=(0, 1, 2),
              num_slices=(10,),
              num_rows=2,
              rescale_method='global',
              background_threshold=0.05,
              annot=None,
              padding=5,
              output_path=None,
              figsize=None, ):
    """
    Color mixer, where each image is represented with a different color (default: red and green) in color channels.

    Parameters
    ----------
    img_spec1 : str or nibabel image-like object
        MR image (or path to one) to be visualized

    img_spec2 : str or nibabel image-like object
        MR image (or path to one) to be visualized

    alpha_channels : (float, float)
        weights for red and green channels in the composite image.
        Default: [1, 1]

    cmap : str or matplotlib.cm.cmap
        Colormap to show the difference values.

    overlay_image : bool
        Flag to specify whether to overlay the first image under the difference map.

    overlay_alpha : float
        Alpha value (to control transparency) for the difference values (to be overlaid on top of the first image).

    cmap : str or matplotlib.cm.cmap
        Colormap to show the difference values.

    overlay_image : bool
        Flag to specify whether to overlay the first image under the difference map.

    overlay_alpha : float
        Alpha value (to control transparency) for the difference values (to be overlaid on top of the first image).

    cmap : str or matplotlib.cm.cmap
        Colormap to show the difference values.

    overlay_image : bool
        Flag to specify whether to overlay the first image under the difference map.

    overlay_alpha : float
        Alpha value (to control transparency) for the difference values (to be overlaid on top of the first image).

    view_set : iterable
        Integers specifying the dimensions to be visualized.
        Choices: one or more of (0, 1, 2) for a 3D image

    num_slices : int or iterable of size as view_set
        number of slices to be selected for each view
        Must be of the same length as view_set,
            each element specifying the number of slices for each dimension.
            If only one number is given, same number will be chosen for all dimensions.

    num_rows : int
        number of rows (top to bottom) per each of 3 dimensions

    rescale_method : bool or str or list or None
        Range to rescale the intensity values to
        Default: 'global', min and max values computed based on ranges from both images.
        If false or None, no rescaling is done (does not work yet).

    background_threshold : float or str
        A threshold value below which all the background voxels will be set to zero.
        Default : 0.05. Other option is a string specifying a percentile: '5%', '10%'.
        Specify None if you don't want any thresholding.

    annot : str
        Text to display to annotate the visualization

    padding : int
        number of voxels to pad around each panel.

    output_path : str
        path to save the generate collage to.

    figsize : list
        Size of figure in inches to be passed on to plt.figure() e.g. [12, 12] or [20, 20]

    Returns
    -------
    fig : figure handle
        handle to the collage figure generated.

    """

    if alpha_channels is None:
        alpha_channels = [1, 1]

    if not len(alpha_channels) == 2:
        # not sure if we should make them sum to 1:  or not np.isclose(sum(alpha_channels), 1.0)
        raise ValueError('Alpha must be two elements')

    mixer_params = dict(alpha_channels=alpha_channels,
                        color_space=color_space)
    fig = _compare(img_spec1,
                   img_spec2,
                   num_rows=num_rows,
                   num_cols=num_slices,
                   mixer='color_mix',
                   annot=annot,
                   padding=padding,
                   rescale_method=rescale_method,
                   bkground_thresh=background_threshold,
                   output_path=output_path,
                   figsize=figsize,
                   **mixer_params)

    return fig


def voxelwise_diff(img_spec1=None,
                   img_spec2=None,
                   abs_value=True,
                   cmap='gray',
                   overlay_image=False,
                   overlay_alpha=0.8,
                   num_rows=2,
                   num_cols=6,
                   rescale_method='global',
                   background_threshold=0.05,
                   annot=None,
                   padding=5,
                   output_path=None,
                   figsize=None):
    """
    Voxel-wise difference map.

    Parameters
    ----------
    img_spec1 : str or nibabel image-like object
        MR image (or path to one) to be visualized

    img_spec2 : str or nibabel image-like object
        MR image (or path to one) to be visualized

    abs_value : bool
        Flag indicating whether to take the absolute value of the diffenence or not.
        Default: True, display absolute differences only (so order of images does not matter)

        Colormap to show the difference values.
    num_rows : int
        number of rows (top to bottom) per each of 3 dimensions

    num_cols : int
        number of panels (left to right) per row of each dimension.

    rescale_method : bool or str or list or None
        Range to rescale the intensity values to
        Default: 'global', min and max values computed based on ranges from both images.
        If false or None, no rescaling is done (does not work yet).

    background_threshold : float or str
        A threshold value below which all the background voxels will be set to zero.
        Default : 0.05. Other option is a string specifying a percentile: '5%', '10%'.
        Specify None if you don't want any thresholding.

    annot : str
        Text to display to annotate the visualization

    padding : int
        number of voxels to pad around each panel.

    output_path : str
        path to save the generate collage to.

    figsize : list
        Size of figure in inches to be passed on to plt.figure() e.g. [12, 12] or [20, 20]

    Returns
    -------
    fig : figure handle
        handle to the collage figure generated.


    """

    if not isinstance(abs_value, bool):
        abs_value = bool(abs_value)

    mixer_params = dict(abs_value=abs_value,
                        cmap=cmap,
                        overlay_image=overlay_image,
                        overlay_alpha=overlay_alpha)
    fig = _compare(img_spec1,
                   img_spec2,
                   num_rows=num_rows,
                   num_cols=num_cols,
                   mixer='voxelwise_diff',
                   annot=annot,
                   padding=padding,
                   rescale_method=rescale_method,
                   bkground_thresh=background_threshold,
                   output_path=output_path,
                   figsize=figsize,
                   **mixer_params)

    return fig


def _compare(img_spec1,
             img_spec2,
             num_rows=2,
             num_cols=6,
             mixer='checker_board',
             rescale_method='global',
             annot=None,
             padding=5,
             bkground_thresh=0.05,
             output_path=None,
             figsize=None,
             **kwargs):
    """
    Produces checkerboard comparison plot of two 3D images.

    Parameters
    ----------
    img_spec1 : str or nibabel image-like object
        MR image (or path to one) to be visualized

    img_spec2 : str or nibabel image-like object
        MR image (or path to one) to be visualized

    num_rows : int
        number of rows (top to bottom) per each of 3 dimensions

    num_cols : int
        number of panels (left to right) per row of each dimension.

    mixer : str
        type of mixer to produce the comparison figure.
        Options: checker_board, color_mix, diff_abs,

    rescale_method : bool or str or list or None
        Method to rescale the intensity values to.
        Choices : 'global', 'each', False or None.

        Default: 'global', min and max values computed based on ranges from both images.
        If 'each', rescales each image separately to [0, 1].
            This option is useful when overlaying images with very different intensity ranges e.g. from different modalities altogether.
        If False or None, no rescaling is done (does not work yet).

    annot : str
        Text to display to annotate the visualization

    padding : int
        number of voxels to pad around each panel.

    output_path : str
        path to save the generate collage to.

    figsize : list
        Size of figure in inches to be passed on to plt.figure() e.g. [12, 12] or [20, 20]

    kwargs : dict
        Additional arguments specific to the particular mixer
        e.g. alpha_channels = [1, 1] for the color_mix mixer

    Returns
    -------

    """

    num_rows, num_cols, padding = check_params(num_rows, num_cols, padding)

    img1, img2 = check_images(img_spec1, img_spec2, bkground_thresh=bkground_thresh)
    img1, img2 = crop_to_extents(img1, img2, padding)

    num_slices_per_view = num_rows * num_cols
    slices = pick_slices(img2, num_slices_per_view)

    rescale_images, img1, img2, min_value, max_value = check_rescaling(img1, img2, rescale_method)

    plt.style.use('dark_background')

    num_axes = 3
    if figsize is None:
        figsize = [3 * num_axes * num_rows, 3 * num_cols]
    fig, ax = plt.subplots(num_axes * num_rows, num_cols, figsize=figsize)

    # displaying some annotation text if provided
    # good choice would be the location of the input images (for future refwhen image is shared or misplaced!)
    if annot is not None:
        fig.suptitle(annot, backgroundcolor='black', color='g')

    display_params = dict(interpolation='none', aspect='equal', origin='lower')

    ax = ax.flatten()
    ax_counter = 0
    for dim_index in range(3):
        for slice_num in slices[dim_index]:
            plt.sca(ax[ax_counter])
            ax_counter = ax_counter + 1

            slice1 = get_axis(img1, dim_index, slice_num)
            slice2 = get_axis(img2, dim_index, slice_num)

            mixed, mixer_spec_params = _generic_mixer(slice1, slice2, mixer, **kwargs)
            display_params.update(mixer_spec_params)

            plt.imshow(mixed, vmin=min_value, vmax=max_value, **display_params)

            # adjustments for proper presentation
            plt.axis('off')

    fig.tight_layout()

    if output_path is not None:
        output_path = output_path.replace(' ', '_')
        fig.savefig(output_path + '.png', bbox_inches='tight')

    # plt.close()

    return fig


def _preprocess_images(img_spec1,
                       img_spec2,
                       rescale_method='global',
                       bkground_thresh=None,
                       padding=5,
                       ):

    img_one, img_two = check_images(img_spec1, img_spec2, bkground_thresh=bkground_thresh)
    img_one, img_two = crop_to_extents(img_one, img_two, padding)

    rescale_images, img_one, img_two, \
    min_value, max_value = check_rescaling(img_one, img_two, rescale_method)

    return img_one, img_two


def _open_figure(slicer, num_rows_per_view, figsize=(15, 11)):

    total_num_rows = len(slicer.view_set) * num_rows_per_view
    total_num_panels = sum(slicer.num_slices)
    num_cols = int(np.ceil(total_num_panels / total_num_rows))

    plt.style.use('dark_background')
    fig, axes = plt.subplots(total_num_rows, num_cols, figsize=figsize,
                             subplot_kw=dict(rasterized=True),
                             gridspec_kw=dict(wspace=0.005, hspace=0.005))
    axes = axes.flatten()
    # plt.subplots_adjust(wspace=0.005, hspace=0.005, top=0.01, bottom=0.0)

    for ix, ax in enumerate(axes):
        ax.axis('off')

    return fig, axes


def collage(img_spec,
            num_rows=2,
            num_cols=6,
            rescale_method='global',
            cmap='gray',
            annot=None,
            padding=5,
            bkground_thresh=None,
            output_path=None,
            figsize=None,
            **kwargs):
    "Produces a collage of various slices from different orientations in the given 3D image"

    num_rows, num_cols, padding = check_params(num_rows, num_cols, padding)

    img = read_image(img_spec, bkground_thresh=bkground_thresh)
    img = crop_image(img, padding)

    img, (min_value, max_value) = check_rescaling_collage(img, rescale_method,
                                                          return_extrema=True)
    num_slices_per_view = num_rows * num_cols
    slices = pick_slices(img, num_slices_per_view)

    plt.style.use('dark_background')

    num_axes = 3
    if figsize is None:
        figsize = [3 * num_axes * num_rows, 3 * num_cols]
    fig, ax = plt.subplots(num_axes * num_rows, num_cols, figsize=figsize)

    # displaying some annotation text if provided
    if annot is not None:
        fig.suptitle(annot, backgroundcolor='black', color='g')

    display_params = dict(interpolation='none', cmap=cmap,
                          aspect='equal', origin='lower',
                          vmin=min_value, vmax=max_value)

    ax = ax.flatten()
    ax_counter = 0
    for dim_index in range(3):
        for slice_num in slices[dim_index]:
            plt.sca(ax[ax_counter])
            ax_counter = ax_counter + 1
            slice1 = get_axis(img, dim_index, slice_num)
            # slice1 = crop_image(slice1, padding)
            plt.imshow(slice1, **display_params)
            plt.axis('off')

    fig.tight_layout()

    if output_path is not None:
        output_path = output_path.replace(' ', '_')
        fig.savefig(output_path + '.png', bbox_inches='tight')

    # plt.close()

    return fig


def aseg_on_mri(mri_spec,
                aseg_spec,
                alpha_mri=1.0,
                alpha_seg=1.0,
                num_rows=2,
                num_cols=6,
                rescale_method='global',
                aseg_cmap='freesurfer',
                sub_cortical=False,
                annot=None,
                padding=5,
                bkground_thresh=0.05,
                output_path=None,
                figsize=None,
                **kwargs):
    "Produces a collage of various slices from different orientations in the given 3D image"

    num_rows, num_cols, padding = check_params(num_rows, num_cols, padding)

    mri = read_image(mri_spec, bkground_thresh=bkground_thresh)
    seg = read_image(aseg_spec, bkground_thresh=0)
    mri, seg = crop_to_seg_extents(mri, seg, padding)

    num_slices_per_view = num_rows * num_cols
    slices = pick_slices(seg, num_slices_per_view)

    plt.style.use('dark_background')

    num_axes = 3
    if figsize is None:
        figsize = [5 * num_axes * num_rows, 5 * num_cols]
    fig, ax = plt.subplots(num_axes * num_rows, num_cols, figsize=figsize)

    # displaying some annotation text if provided
    if annot is not None:
        fig.suptitle(annot, backgroundcolor='black', color='g')

    display_params_mri = dict(interpolation='none', aspect='equal', origin='lower',
                              cmap='gray', alpha=alpha_mri,
                              vmin=mri.min(), vmax=mri.max())
    display_params_seg = dict(interpolation='none', aspect='equal', origin='lower',
                              alpha=alpha_seg)

    normalize_labels = colors.Normalize(vmin=seg.min(), vmax=seg.max(), clip=True)
    fs_cmap = get_freesurfer_cmap(sub_cortical)
    label_mapper = cm.ScalarMappable(norm=normalize_labels, cmap=fs_cmap)

    ax = ax.flatten()
    ax_counter = 0
    for dim_index in range(3):
        for slice_num in slices[dim_index]:
            plt.sca(ax[ax_counter])
            ax_counter = ax_counter + 1

            slice_mri = get_axis(mri, dim_index, slice_num)
            slice_seg = get_axis(seg, dim_index, slice_num)

            # # masking data to set no-value pixels to transparent
            # seg_background = np.isclose(slice_seg, 0.0)
            # slice_seg = np.ma.masked_where(seg_background, slice_seg)
            # slice_mri = np.ma.masked_where(np.logical_not(seg_background), slice_mri)

            seg_rgb = label_mapper.to_rgba(slice_seg)
            plt.imshow(seg_rgb, **display_params_seg)
            plt.imshow(slice_mri, **display_params_mri)
            plt.axis('off')

    # plt.subplots_adjust(wspace=0.0, hspace=0.0)
    plt.subplots_adjust(left=0.01, right=0.99,
                        bottom=0.01, top=0.99,
                        wspace=0.05, hspace=0.02)
    # fig.tight_layout()

    if output_path is not None:
        output_path = output_path.replace(' ', '_')
        fig.savefig(output_path + '.png', bbox_inches='tight')

    # plt.close()

    return fig


def _generic_mixer(slice1, slice2, mixer_name, **kwargs):
    """
    Generic mixer to process two slices with appropriate mixer
        and return the composite to be displayed.
    """

    mixer_name = mixer_name.lower()
    if mixer_name in ['color_mix', 'rgb']:
        mixed = _mix_color(slice1, slice2, **kwargs)
        cmap = None  # data is already RGB-ed
    elif mixer_name in ['checkerboard', 'checker', 'cb', 'checker_board']:
        checkers = _get_checkers(slice1.shape, **kwargs)
        mixed = _checker_mixer(slice1, slice2, checkers)
        cmap = 'gray'
    elif mixer_name in ['diff', 'voxelwise_diff', 'vdiff']:

        mixed, cmap = _diff_image(slice1, slice2, **kwargs)
        # if kwargs['overlay_image'] is True:
        #     diff_cmap = diff_colormap()
        #     plt.imshow(slice1, alpha=kwargs['overlay_alpha'], **display_params)
        #     plt.hold(True)
        #     plt.imshow(mixed,
        #                cmap=diff_cmap,
        #                vmin=min_value, vmax=max_value,
        #                **display_params)
        # else:
        #     plt.imshow(mixed, cmap=cmap,
        #                vmin=min_value, vmax=max_value,
        #                **display_params)

    else:
        raise ValueError('Invalid mixer name chosen.')

    disp_params = dict(cmap=cmap)

    return mixed, disp_params


def check_rescaling_collage(img, rescale_method=None,
                            return_extrema=True):
    ""

    if not (isinstance(rescale_method, str) or rescale_method is None):
        raise ValueError('rescale_method method be "global", "slice" or None')

    if rescale_method is None:
        return img, rescale_method

    rescale_method = rescale_method.lower()
    if rescale_method in ['global']:
        img = scale_0to1(img)

    if return_extrema:
        min_value = img.min()
        max_value = img.max()
        return img, (min_value, max_value)
    else:
        return img


def check_rescaling(img1, img2, rescale_method):
    """Estimates the intensity range to clip the visualizations to"""

    # estimating intensity ranges
    if rescale_method is None:
        # this section is to help user to avoid all intensity rescaling altogther!
        # TODO bug does not work yet, as pyplot does not offer any easy way to control it
        rescale_images = False
        min_value = None
        max_value = None
        norm_image = None  # mpl.colors.NoNorm doesn't work yet. data is getting linearly normalized to [0, 1]
    elif isinstance(rescale_method, str):
        if rescale_method.lower() in ['global']:
            # TODO need a way to alert the user if one of the distributions is too narrow
            #  in which case that image will be collapsed to an uniform value
            combined_distr = np.concatenate((img1.flatten(), img2.flatten()))
            min_value = combined_distr.min()
            max_value = combined_distr.max()
        elif rescale_method.lower() in ['each']:
            img1 = scale_0to1(img1)
            img2 = scale_0to1(img2)
            min_value = 0.0
            max_value = 1.0
        else:
            raise ValueError('rescaling method can only be "global" or "each"')

        rescale_images = True
        norm_image = mpl.colors.Normalize
    elif len(rescale_method) == 2:
        min_value = min(rescale_method)
        max_value = max(rescale_method)
        rescale_images = True
        norm_image = mpl.colors.Normalize
    else:
        raise ValueError('Invalid intensity range!. It must be either : '
                         '1) a list/tuple of two distinct values or'
                         '2) "global" indicating rescaling based on min/max values derived from both images or'
                         '3) None, no rescaling or norming altogether. ')

    return rescale_images, img1, img2, min_value, max_value


def check_images(img_spec1, img_spec2, bkground_thresh=0.05):
    """Reads the two images and assers identical shape."""

    img1 = read_image(img_spec1, bkground_thresh)
    img2 = read_image(img_spec2, bkground_thresh)

    if img1.shape != img2.shape:
        raise ValueError('size mismatch! First image: {} Second image: {}\n'
                         'Two images to be compared must be of the same size in all dimensions.'.format(
            img1.shape,
            img2.shape))

    return img1, img2


def _get_checkers(slice_shape, patch_size):
    """Creates checkerboard of a given tile size, filling a given slice."""

    if patch_size is not None:
        patch_size = check_patch_size(patch_size)
    else:
        # 7 patches in each axis, min voxels/patch = 3
        # TODO make 7 a user settable parameter
        patch_size = np.round(np.array(slice_shape) / 7).astype('int16')
        patch_size = np.maximum(patch_size, np.array([3, 3]))

    black = np.zeros(patch_size)
    white = np.ones(patch_size)
    tile = np.vstack((np.hstack([black, white]), np.hstack([white, black])))

    # using ceil so we can clip the extra portions
    num_tiles = np.ceil(np.divide(slice_shape, tile.shape)).astype(int)
    checkers = np.tile(tile, num_tiles)

    # clipping any extra columns or rows
    if any(np.greater(checkers.shape, slice_shape)):
        if checkers.shape[0] > slice_shape[0]:
            checkers = np.delete(checkers, np.s_[slice_shape[0]:], axis=0)
        if checkers.shape[1] > slice_shape[1]:
            checkers = np.delete(checkers, np.s_[slice_shape[1]:], axis=1)

    return checkers


def _mix_color(slice1, slice2, alpha_channels, color_space):
    """Mixing them as red and green channels"""

    if slice1.shape != slice2.shape:
        raise ValueError('size mismatch between cropped slices and checkers!!!')

    alpha_channels = np.array(alpha_channels)
    if len(alpha_channels) != 2:
        raise ValueError('Alphas must be two value tuples.')

    slice1, slice2 = scale_images_0to1(slice1, slice2)

    # masking background
    combined_distr = np.concatenate((slice1.flatten(), slice2.flatten()))
    image_eps = np.percentile(combined_distr, 5)
    background = np.logical_or(slice1 <= image_eps, slice2 <= image_eps)

    if color_space.lower() in ['rgb']:

        red = alpha_channels[0] * slice1
        grn = alpha_channels[1] * slice2
        blu = np.zeros_like(slice1)

        # foreground = np.logical_not(background)
        # blu[foreground] = 1.0

        mixed = np.stack((red, grn, blu), axis=2)

    elif color_space.lower() in ['hsv']:

        raise NotImplementedError(
            'This method (color_space="hsv") is yet to fully conceptualized and implemented.')

        # TODO other ideas: hue/saturation/intensity value driven by difference in intensity?
        hue = alpha_channels[0] * slice1
        sat = alpha_channels[1] * slice2
        val = np.ones_like(slice1)

        hue[background] = 1.0
        sat[background] = 0.0
        val[background] = 0.0

        mixed = np.stack((hue, sat, val), axis=2)
        # converting to RGB
        mixed = mpl.colors.hsv_to_rgb(mixed)

    # ensuring all values are clipped to [0, 1]
    mixed[mixed <= 0.0] = 0.0
    mixed[mixed >= 1.0] = 1.0

    return mixed


def _checker_mixer(slice1,
                   slice2,
                   checker_size=None):
    """Mixes the two slices in alternating areas specified by checkers"""

    checkers = _get_checkers(slice1.shape, checker_size)
    if slice1.shape != slice2.shape or slice2.shape != checkers.shape:
        raise ValueError('size mismatch between cropped slices and checkers!!!')

    mixed = slice1.copy()
    mixed[checkers > 0] = slice2[checkers > 0]

    return mixed


def cli_run():
    raise NotImplementedError


if __name__ == '__main__':
    cli_run()
