"""
mrivis: Tools to comapre the similarity of two 3d images (structural, functional or parametric maps)

Options include checker board, red green mixer and voxel-wise difference maps.

"""

__all__ = ['checkerboard', 'color_mix', 'voxelwise_diff']

import numpy as np
from matplotlib import pyplot as plt
import nibabel as nib
from genericpath import exists as pexists
from os.path import realpath


def checkerboard(img_spec1=None,
                 img_spec2=None,
                 patch_size=10,
                 num_rows=2,
                 num_cols=6,
                 rescale_intensity_range=None,
                 annot=None,
                 padding=5,
                 output_path=None,
                 figsize=None,):
    """
    Checkerboard mixer.

    Parameters
    ----------
    img_spec1 : str or nibabel image-like object
        MR image (or path to one) to be visualized

    img_spec2 : str or nibabel image-like object
        MR image (or path to one) to be visualized

    patch_size : int, or list, (int, int)
        size of checker patch (either square or rectangular)

    num_rows : int
        number of rows (top to bottom) per each of 3 dimensions

    num_cols : int
        number of panels (left to right) per row of each dimension.

    rescale_intensity_range : bool or list
        range to rescale the intensity values to

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

    mixer_params = dict(patch_size=patch_size)

    fig = _compare(img_spec1,
                   img_spec2,
                   num_rows=num_rows,
                   num_cols=num_cols,
                   mixer='checker_board',
                   rescale_intensity_range=rescale_intensity_range,
                   annot=annot,
                   padding=padding,
                   output_path=output_path,
                   figsize=figsize,
                   **mixer_params)

    return fig


def color_mix(img_spec1=None,
              img_spec2=None,
              alpha_channels=None,
              num_rows=2,
              num_cols=6,
              rescale_intensity_range=None,
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

    num_rows : int
        number of rows (top to bottom) per each of 3 dimensions

    num_cols : int
        number of panels (left to right) per row of each dimension.

    rescale_intensity_range : bool or list
        range to rescale the intensity values to

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

    mixer_params = dict(alpha_channels=alpha_channels)
    fig = _compare(img_spec1,
                   img_spec2,
                   num_rows=num_rows,
                   num_cols=num_cols,
                   mixer='color_mix',
                   annot=annot,
                   padding=padding,
                   output_path=output_path,
                   figsize=figsize,
                   **mixer_params)

    return fig


def voxelwise_diff(img_spec1=None,
                   img_spec2=None,
                   abs_value=True,
                   num_rows=2,
                   num_cols=6,
                   rescale_intensity_range=None,
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

    num_rows : int
        number of rows (top to bottom) per each of 3 dimensions

    num_cols : int
        number of panels (left to right) per row of each dimension.

    rescale_intensity_range : bool or list
        range to rescale the intensity values to

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

    mixer_params = dict(abs_value=abs_value)
    fig = _compare(img_spec1,
                   img_spec2,
                   num_rows=num_rows,
                   num_cols=num_cols,
                   mixer='voxelwise_diff',
                   annot=annot,
                   padding=padding,
                   output_path=output_path,
                   figsize=figsize,
                   **mixer_params)

    return fig


def _compare(img_spec1,
             img_spec2,
             num_rows=2,
             num_cols=6,
             mixer='checker_board',
             rescale_intensity_range=None,
             annot=None,
             padding=5,
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

    rescale_intensity_range : bool or list
        range to rescale the intensity values to

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

    img1, img2 = check_images(img_spec1, img_spec2)
    img1, img2 = crop_to_extents(img1, img2, padding)

    slices = pick_slices(img1.shape, num_rows, num_cols)

    rescale_images, img_intensity_range = check_rescaling(img1, rescale_intensity_range)

    plt.style.use('dark_background')

    num_axes = 3
    if figsize is None:
        figsize = [15, 15]
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

            if rescale_images:
                plt.imshow(mixed, imlim=img_intensity_range, **display_params)
            else:
                plt.imshow(mixed, **display_params)

            # adjustments for proper presentation
            plt.axis('off')

    fig.tight_layout()

    if output_path is not None:
        output_path = output_path.replace(' ', '_')
        fig.savefig(output_path + '.png', bbox_inches='tight')

    # plt.close()

    return fig


def _generic_mixer(slice1, slice2, mixer_name, **kwargs):
    """Generic mixer to process two slices with appropriate mixer and return the composite to be displayed."""

    mixer_name = mixer_name.lower()
    if mixer_name in ['color_mix', 'rgb']:
        mixed = _mix_color(slice1, slice2, **kwargs)
        cmap = 'gray'
    elif mixer_name in ['checkerboard', 'checker', 'cb', 'checker_board']:
        checkers = _get_checkers(slice1.shape, **kwargs)
        mixed = _mix_slices_in_checkers(slice1, slice2, checkers)
        cmap = 'gray'
    elif mixer_name in ['diff', 'voxelwise_diff', 'vdiff']:
        mixed = _diff_image(slice1, slice2, **kwargs)
        cmap = 'gray'
    else:
        raise ValueError('Invalid mixer name chosen.')

    disp_params = dict(cmap=cmap)

    return mixed, disp_params


def _diff_image(slice1, slice2, abs_value=True):
    """Computes the difference image"""

    diff = slice1-slice2

    if abs_value:
        diff = np.abs(diff)

    return diff


def get_axis(array, axis, slice_num):
    """Returns a fixed axis"""

    slice_list = [slice(None)] * array.ndim
    slice_list[axis] = slice_num
    slice_data = array[slice_list].T  # transpose for proper orientation

    return slice_data


def check_int(num, num_descr):
    """Validation and typecasting."""

    if not np.isfinite(num) or num < 1:
        raise ValueError('{} is not finite or is not > 0'.format(num_descr))

    return int(num)


def check_patch_size(patch_size):
    """Validation and typcasting"""

    patch_size = np.array(patch_size)
    if patch_size.size == 1:
        patch_size = np.repeat(patch_size, 2).astype('int16')

    return patch_size


def check_params(num_rows, num_cols, padding):
    """Validation and typcasting"""

    num_rows = check_int(num_rows, 'num_rows')
    num_cols = check_int(num_cols, 'num_cols')
    padding = check_int(padding, 'padding')

    return num_rows, num_cols, padding


def pick_slices(img_shape, num_rows, num_cols):
    """Picks the slices to display in each dimension"""

    num_panels = num_rows * num_cols * 3
    skip_count = num_rows * num_cols

    slices = list()
    for dim_size in img_shape:
        slices_in_dim = np.around(np.linspace(0, dim_size, num_panels)).astype('int64')
        # skipping not-so-important slices at boundaries
        slices_in_dim = slices_in_dim[skip_count: -skip_count]
        slices.append(slices_in_dim)

    return slices


def check_rescaling(img1, rescale_intensity_range):
    """Estimates the intensity range to clip the visualizations to"""

    RescaleImages = True
    # estimating intensity ranges
    if rescale_intensity_range is None:
        img_intensity_range = [img1.min(), img1.max()]
    elif len(rescale_intensity_range) == 2:
        img_intensity_range = rescale_intensity_range
    else:
        RescaleImages = False
        img_intensity_range = None

    if len(np.unique(img_intensity_range)) == 1:
        RescaleImages = False

    return RescaleImages, img_intensity_range


def check_images(img_spec1, img_spec2):
    """Reads the two images and assers identical shape."""

    img1 = read_image(img_spec1)
    img2 = read_image(img_spec2)

    if img1.shape != img2.shape:
        raise ValueError('size mismatch! Two images to be compared must be of the same size in all dimensions.')

    return img1, img2


def read_image(img_spec):
    """Image reader. Removes stray values close to zero (smaller than 5 %ile)."""

    if isinstance(img_spec, str):
        if pexists(realpath(img_spec)):
            img = nib.load(img_spec).get_data()
        else:
            raise IOError('Given path to image does not exist!')
    elif isinstance(img_spec, np.ndarray):
        img = img_spec
    else:
        raise ValueError('Invalid input specified! '
                         'Input either a path to image data, or provide 3d Matrix directly.')

    if len(img.shape) < 3:
        raise ValueError('Input volume must be atleast 3D!')
    elif len(img.shape) == 3:
        for dim_size in img.shape:
            if dim_size < 1:
                raise ValueError('Atleast one slice must exist in each dimension')
    elif len(img.shape) == 4:
        if img.shape[3] != 1:
            raise ValueError('Input volume is 4D with more than one volume!')
        else:
            img = np.squeeze(img, axis=3)
    elif len(img.shape) > 4:
        raise ValueError('Invalid shape of image : {}'.format(img.shape))

    # setting small stray values to zero
    stray_value = np.percentile(img.flatten(), 5)
    img[img < stray_value] = 0.0

    return img


def _get_checkers(slice_shape, patch_size):
    """Creates checkerboard of a given tile size, filling a given slice."""

    patch_size = check_patch_size(patch_size)

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


def scale_0to1(slice1, slice2):
    """Scale the two images to [0, 1] based on min/max from both."""

    min_value = max(slice1.min(), slice2.min())
    max_value = max(slice1.max(), slice2.max())

    slice1 = (slice1 - min_value) / max_value
    slice2 = (slice2 - min_value) / max_value

    return slice1, slice2


def _mix_color(slice1, slice2, alpha_channels):
    """Mixing them as red and green channels"""

    if slice1.shape != slice2.shape:
        raise ValueError('size mismatch between cropped slices and checkers!!!')

    alpha_channels = np.array(alpha_channels)
    if len(alpha_channels) != 2:
        raise ValueError('Alphas must be two value tuples.')

    slice1, slice2 = scale_0to1(slice1, slice2)

    mixed = np.zeros((slice1.shape[0], slice1.shape[1], 3))
    mixed[:, :, 0] = alpha_channels[0] * slice1
    mixed[:, :, 1] = alpha_channels[1] * slice2
    mixed[:, :, 2] = 0.0  # leaving blue as zero

    return mixed


def _mix_slices_in_checkers(slice1, slice2, checkers):
    """Mixes the two slices in alternating areas specified by checkers"""

    if slice1.shape != slice2.shape or slice2.shape != checkers.shape:
        raise ValueError('size mismatch between cropped slices and checkers!!!')

    mixed = slice1.copy()
    mixed[checkers > 0] = slice2[checkers > 0]

    return mixed


def crop_to_extents(img1, img2, padding):
    """Crop the images to ensure both fit within the bounding box"""

    beg_coords1, end_coords1 = crop_coords(img1, padding)
    beg_coords2, end_coords2 = crop_coords(img2, padding)

    beg_coords = np.fmin(beg_coords1, beg_coords2)
    end_coords = np.fmax(end_coords1, end_coords2)

    img1 = crop_3dimage(img1, beg_coords, end_coords)
    img2 = crop_3dimage(img2, beg_coords, end_coords)

    return img1, img2


def crop_coords(img, padding):
    """Find coordinates describing extent of non-zero portion of image, padded"""

    coords = np.nonzero(img)
    empty_axis_exists = np.any([len(arr) == 0 for arr in coords])
    if empty_axis_exists:
        end_coords = img.shape
        beg_coords = np.ones((0, img.ndim)).astype(int)
    else:
        min_coords = np.array([arr.min() for arr in coords])
        max_coords = np.array([arr.max() for arr in coords])
        beg_coords = np.fmax(0, min_coords - padding)
        end_coords = np.fmin(img.shape, max_coords + padding)

    return beg_coords, end_coords


def crop_3dimage(img, beg_coords, end_coords):
    """Crops a 3d image to the bounding box specified."""

    cropped_img = img[
                  beg_coords[0]:end_coords[0],
                  beg_coords[1]:end_coords[1],
                  beg_coords[2]:end_coords[2]
                  ]

    return cropped_img


def cli_run():
    raise NotImplementedError


if __name__ == '__main__':
    cli_run()
