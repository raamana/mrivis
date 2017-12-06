
__all__ = ['checkerboard']

import numpy as np
from matplotlib import pyplot as plt
import nibabel as nib
from genericpath import exists as pexists
from os.path import realpath

def checkerboard(img_spec1=None,
                 img_spec2=None,
                 patch_size=10,
                 num_rows = 2,
                 num_cols = 6,
                 rescale_intensity_range=None,
                 annot=None,
                 padding=5):
    """
    Produces checkerboard comparison plot of two 3D images.

    Parameters
    ----------
    img_spec1 : str or nibabel image-like object
        MR image (or path to one) to be visualized

    img_spec2 : str or nibabel image-like object
        MR image (or path to one) to be visualized

    patch_size : int, or (int, int)
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

    Returns
    -------

    """

    patch_size, num_rows, num_cols, padding = check_params(patch_size, num_rows, num_cols, padding)
    img1, img2 = check_images(img_spec1, img_spec2)
    img1, img2 = crop_to_extents(img1, img2, padding)

    num_panels = num_rows*num_cols*3
    # skipping few at first and last
    skip_count = num_rows*num_cols

    slices = list()
    for dim_size in img1.shape:
        slices_in_dim = np.around(np.linspace(0, dim_size, num_panels)).astype('int64')
        slices.append(slices_in_dim)

    RescaleImages = True
    # estimating intensity ranges
    if rescale_intensity_range is None:
        img_intensity_range = [img1.min(), img1.max()]
    elif len(rescale_intensity_range) == 2:
        img_intensity_range = rescale_intensity_range
    else:
        RescaleImages = False

    if len(np.unique(img_intensity_range)) == 1:
        RescaleImages = False

    plt.style.use('dark_background')

    num_axes = 3
    fig, ax = plt.subplots(num_axes*num_rows, num_cols, figsize=[15, 15])
    ax = ax.flatten()

    ax_counter = 0
    for dim_index in range(3):
        slices_this_dim = slices[dim_index][skip_count : -skip_count]

        for slice_num in slices_this_dim:
            plt.sca(ax[ax_counter])

            slice1 = get_axis(img1, dim_index, slice_num)
            slice2 = get_axis(img2, dim_index, slice_num)

            checkers = get_checkers(slice1.shape, patch_size)
            mixed = mix_slices(slice1, slice2, checkers)

            if RescaleImages:
                plt.imshow(mixed.T, imlim=img_intensity_range,
                           aspect='equal', origin='lower')
            else:
                plt.imshow(mixed.T,
                           aspect='equal', origin='lower')

            # adjustments for proper presentation
            plt.set_cmap('gray')
            plt.axis('off')

            ax[ax_counter].set_title(ax_counter)
            ax_counter = ax_counter + 1

    # displaying some annotation text if provided
    # good choice would be the location of the input image (for future refwhen image is shared or misplaced!)
    if annot is not None:
        plt.text(0.05, 0.5, annot, backgroundcolor='black', color='g')

    fig.tight_layout()

    return


def get_axis(array, axis, slice_num):
    "Returns a fixed axis"

    slice_list = [slice(None)] * array.ndim
    slice_list[axis] = slice_num

    return array[slice_list]


def check_int(num, num_descr):

    if not np.isfinite(num) or num<1:
        raise ValueError('{} is not finite or is not > 0'.format(num_descr))

    return int(num)


def check_params(patch_size, num_rows, num_cols, padding):
    "Validation and typcasting"

    patch_size = np.array(patch_size)
    if patch_size.size == 1:
        patch_size = np.repeat(patch_size, 2)

    num_rows = check_int(num_rows, 'num_rows')
    num_cols = check_int(num_cols, 'num_cols')
    padding = check_int(padding, 'padding')

    return patch_size, num_rows, num_cols, padding


def check_images(img_spec1, img_spec2):

    img1 = read_image(img_spec1)
    img2 = read_image(img_spec2)

    if img1.shape != img2.shape:
        raise ValueError('size mismatch! Two images to be compared must be of the same size in all dimensions.')

    return img1, img2


def read_image(img_spec):
    # reading in data

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

    return img


def get_checkers(slice_shape, patch_size):
    " Creates checkerboard of a given tile size, filling a given slice."

    black = np.zeros(patch_size)
    white = np.ones(patch_size)
    tile = np.vstack((np.hstack([black, white]), np.hstack([white, black])))

    # using ceil so we can clip the extra portions
    num_tiles = np.ceil(np.divide(slice_shape, tile.shape)).astype(int)
    checkers  = np.tile(tile, num_tiles)

    # clipping any extra columns or rows
    if any(np.greater(checkers.shape, slice_shape)):
        if checkers.shape[0] > slice_shape[0]:
            checkers = np.delete(checkers, np.s_[slice_shape[0]:], axis=0)
        if checkers.shape[1] > slice_shape[1]:
            checkers = np.delete(checkers, np.s_[slice_shape[1]:], axis=1)

    return checkers


def mix_slices(slice1, slice2, checkers):
    "Mixes the two slices in alternating areas specified by checkers"

    mixed = slice1.copy()
    mixed[checkers > 0] = slice2[checkers > 0]

    return mixed


def crop_to_extents(img1, img2, padding):
    "Crop the images to ensure both fit within the bounding box"

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
        beg_coords = np.fmax(0        , min_coords - padding)
        end_coords = np.fmin(img.shape, max_coords + padding)

    return beg_coords, end_coords


def crop_3dimage(img, beg_coords, end_coords):

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
