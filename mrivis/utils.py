from collections.abc import Iterable
from os.path import exists as pexists, realpath

import matplotlib as mpl
import nibabel as nib
import numpy as np
from matplotlib import colormaps, pyplot as plt


def _diff_image(slice1, slice2,
                abs_value=True,
                cmap='gray',
                **kwargs):
    """Computes the difference image"""

    diff = slice1 - slice2

    if abs_value:
        diff = np.abs(diff)

    return diff, cmap


def diff_colormap():
    """Custom colormap to map low values to black or another color."""

    # bottom = colormaps['copper'](np.linspace(0., 1, 6))
    black = np.atleast_2d([0., 0., 0., 1.])
    bottom = np.repeat(black, 6, axis=0)
    middle = colormaps['copper'](np.linspace(0, 1, 250))
    # remain = plt.cm.Reds(np.linspace(0, 1, 240))

    colors = np.vstack((bottom, middle))
    diff_cmap = mpl.colors.LinearSegmentedColormap.from_list('diff_colormap', colors)

    return diff_cmap


def get_axis(array, axis, slice_num):
    """Returns a fixed axis"""

    slice_list = [slice(None)] * array.ndim
    slice_list[axis] = slice_num
    slice_data = array[tuple(slice_list)].T  # transpose for proper orientation

    return slice_data


def check_bounding_rect(rect_pos):
    """Ensure the rect spec is valid."""

    if not isinstance(rect_pos, Iterable):
        raise ValueError('rectangle spec must be a tuple of floats '
                         'specifying (left, right, width, height)')

    left, bottom, width, height = rect_pos
    for val, name in zip((left, bottom, width, height),
                         ('left', 'bottom', 'width', 'height')):
        if val < 0.0 or val > 1.0:
            raise ValueError(f"{name}'s value must be >=0 and <= 1.0. "
                             f"It is now {val}")

    if left + width > 1.0:
        print(f'rect would extend beyond the width of figure/axis by {left + width - 1.0}')

    if bottom + height > 1.0:
        print(f'rect would extend beyond the height of figure/axis by {bottom + height - 1.0}')

    return rect_pos


def check_views(view_set, max_views=3):
    """Ensures valid view/dimensions are selected."""

    if not isinstance(view_set, Iterable):
        view_set = tuple([view_set, ])

    if len(view_set) > max_views:
        raise ValueError(f'Can only have {max_views} views')

    return [check_int(view, 'view', min_value=0, max_value=max_views - 1) for view in view_set]


def check_num_slices(num_slices, img_shape=None, num_dims=3):
    """Ensures requested number of slices is valid.

    At least 1 and at most the image size, if available
    """

    if not isinstance(num_slices, Iterable) or len(num_slices) == 1:
        num_slices = np.repeat(num_slices, num_dims)

    if img_shape is not None:
        if len(num_slices) != len(img_shape):
            raise ValueError('The number of dimensions requested is different from image.'
                             f' Must be either 1 or equal to {len(img_shape) + 1}')
        # upper bounding them to image shape
        num_slices = np.minimum(img_shape, num_slices)

    # lower bounding it to 1
    return np.maximum(1, num_slices)


def check_int(num,
              num_descr='number',
              min_value=0,
              max_value=np.inf):
    """Validation and typecasting."""

    if not np.isfinite(num) or num < min_value or num > max_value:
        raise ValueError(f'{num_descr}={num} is not finite or '
                         f'is not >= {min_value} or '
                         f'is not < {max_value}')

    return int(num)


def check_patch_size(patch_size):
    """Validation and typecasting"""

    patch_size = np.array(patch_size)
    if patch_size.size == 1:
        patch_size = np.repeat(patch_size, 2).astype('int16')

    return patch_size


def check_params(num_rows, num_cols, padding):
    """Validation and typecasting"""

    num_rows = check_int(num_rows, 'num_rows', min_value=1)
    num_cols = check_int(num_cols, 'num_cols', min_value=1)
    padding = check_int(padding, 'padding', min_value=0)

    return num_rows, num_cols, padding


def read_image(img_spec, bkground_thresh, ensure_num_dim=3):
    """Image reader, with additional checks on size.

    Can optionally remove stray values close to zero (smaller than 5 %ile)."""

    img = load_image_from_disk(img_spec)

    if not np.issubdtype(img.dtype, np.floating):
        img = img.astype('float32')

    if ensure_num_dim == 3:
        img = check_image_is_3d(img)
    elif ensure_num_dim == 4:
        img = check_image_is_4d(img)

    return threshold_image(img, bkground_thresh)


def load_image_from_disk(img_spec):
    """Vanilla image loader."""

    if isinstance(img_spec, str):
        if pexists(realpath(img_spec)):
            hdr = nib.load(img_spec)
            # trying to stick to an orientation
            hdr = nib.as_closest_canonical(hdr)
            img = hdr.get_fdata()
        else:
            raise IOError('Given path to image does not exist!')
    elif isinstance(img_spec, np.ndarray):
        img = img_spec
    else:
        raise ValueError('Invalid input specified! '
                         'Input either a path to image data, or provide 3d Matrix directly.')

    return img


def check_matching_dims(img_one, img_two):
    """Checks if the dimensions of the two images match, excluding singleton dims."""

    return np.isclose(np.squeeze(img_one.shape), np.squeeze(img_two.shape)).all()


def check_image_is_3d(img):
    """Ensures the image loaded is 3d and nothing else."""

    if len(img.shape) < 3:
        raise ValueError('Input volume must be at least 3D!')
    elif len(img.shape) == 3:
        for dim_size in img.shape:
            if dim_size < 1:
                raise ValueError('At least one slice must exist in each dimension')
    elif len(img.shape) == 4:
        if img.shape[3] != 1:
            raise ValueError('Input volume is 4D with more than one volume!')
        else:
            img = np.squeeze(img, axis=3)
    elif len(img.shape) > 4:
        raise ValueError(f'Invalid shape of image : {img.shape}')

    return img


def check_image_is_4d(img, min_num_volumes=2):
    """Ensures the image loaded is 3d and nothing else."""

    if len(img.shape) < 4:
        raise ValueError('Input volume must be 4D!')
    elif len(img.shape) == 4:
        for dim_size in img.shape[:3]:
            if dim_size < 1:
                raise ValueError('At least one slice must exist in each dimension')
        if img.shape[3] < min_num_volumes:
            raise ValueError('Input volume is 4D '
                             f'with less than {min_num_volumes} volumes!')
    elif len(img.shape) > 4:
        raise ValueError('Too many dimensions : more than 4.\n'
                         f'Invalid shape of image : {img.shape}')

    return img


def threshold_image(img, bkground_thresh, bkground_value=0.0):
    """
    Thresholds a given image at a value or percentile.

    Replacement value can be specified too.


    Parameters
    -----------
    img : ndarray
        Input image

    bkground_thresh : float
        a threshold value to identify the background

    bkground_value : float
        a value to fill the background elements with. Default 0.

    Returns
    -------

    thresholded_image : ndarray
        thresholded and/or filled image

    """

    if bkground_thresh is None:
        return img

    if isinstance(bkground_thresh, str):
        try:
            thresh_perc = float(bkground_thresh.replace('%', ''))
        except:
            raise ValueError(
                'percentile specified could not be parsed correctly '
                ' - must be a string of the form "5%", "10%" etc')
        else:
            thresh_value = np.percentile(img, thresh_perc)
    elif isinstance(bkground_thresh, (float, int)):
        thresh_value = bkground_thresh
    else:
        raise ValueError('Invalid specification for background threshold.')

    img[img < thresh_value] = bkground_value

    return img


def scale_0to1(image_in,
               exclude_outliers_below=False,
               exclude_outliers_above=False):
    """Scale the two images to [0, 1] based on min/max from both.

    Parameters
    -----------
    image_in : ndarray
        Input image

    exclude_outliers_{below,above} : float
        Lower/upper limit, a value between 0 and 100.

    Returns
    -------

    scaled_image : ndarray
        clipped and/or scaled image

    """

    min_value = image_in.min()
    max_value = image_in.max()
    # making a copy to ensure no side effects
    image = image_in.copy()
    if exclude_outliers_below:
        perctl = float(exclude_outliers_below)
        image[image < np.percentile(image, perctl)] = min_value

    if exclude_outliers_above:
        perctl = float(exclude_outliers_above)
        image[image > np.percentile(image, 100.0 - perctl)] = max_value

    image = (image - min_value) / (max_value - min_value)

    return image


def row_wise_rescale(matrix):
    """
    Row-wise rescale of a given matrix.

    For fMRI data (num_voxels x num_time_points), this would translate to voxel-wise normalization over time.

    Parameters
    ----------

    matrix : ndarray
        Input rectangular matrix, typically a carpet of size num_voxels x num_4th_dim, 4th_dim could be time points or gradients or other appropriate

    Returns
    -------
    normed : ndarray
        normalized matrix

    """

    if matrix.shape[0] <= matrix.shape[1]:
        raise ValueError('Number of voxels is less than the number of time points!! '
                         'Are you sure data is reshaped correctly?')

    min_ = matrix.min(axis=1)
    range_ = np.ptp(matrix, axis=1)  # ptp : peak to peak, max-min
    min_tile = np.tile(min_, (matrix.shape[1], 1)).T
    range_tile = np.tile(range_, (matrix.shape[1], 1)).T
    # avoiding any numerical difficulties
    range_tile[range_tile < np.finfo(float).eps] = 1.0

    normed = (matrix - min_tile) / range_tile

    del min_, range_, min_tile, range_tile

    return normed


def crop_to_seg_extents(img, seg, padding):
    """Crop the image (usually MRI) to fit within the bounding box of a segmentation (or set of seg)"""

    beg_coords, end_coords = crop_coords(seg, padding)

    img = crop_3dimage(img, beg_coords, end_coords)
    seg = crop_3dimage(seg, beg_coords, end_coords)

    return img, seg


def crop_to_extents(img1, img2, padding):
    """Crop the images to ensure both fit within the bounding box"""

    beg_coords1, end_coords1 = crop_coords(img1, padding)
    beg_coords2, end_coords2 = crop_coords(img2, padding)

    beg_coords = np.fmin(beg_coords1, beg_coords2)
    end_coords = np.fmax(end_coords1, end_coords2)

    img1 = crop_3dimage(img1, beg_coords, end_coords)
    img2 = crop_3dimage(img2, beg_coords, end_coords)

    return img1, img2


def crop_image(img, padding=5):
    """Crops an image or slice to its extents"""

    if padding < 1:
        return img

    beg_coords, end_coords = crop_coords(img, padding)

    if len(img.shape) == 3:
        img = crop_3dimage(img, beg_coords, end_coords)
    elif len(img.shape) == 2:
        img = crop_2dimage(img, beg_coords, end_coords)
    else:
        raise ValueError('Can only crop 2D or 3D images!')

    return img


def crop_coords(img, padding):
    """Find coordinates describing extent of non-zero portion of image, padded"""

    coords = np.nonzero(img)
    empty_axis_exists = np.any([len(arr) == 0 for arr in coords])
    if empty_axis_exists:
        end_coords = img.shape
        beg_coords = np.zeros((1, img.ndim)).astype(int).flatten()
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


def crop_2dimage(img, beg_coords, end_coords):
    """Crops a 3d image to the bounding box specified."""

    cropped_img = img[
                  beg_coords[0]:end_coords[0],
                  beg_coords[1]:end_coords[1],
                  ]

    return cropped_img


def pick_slices(img, num_slices_per_view):
    """
    Picks the slices to display in each dimension,
        skipping any empty slices (without any segmentation at all).

    """

    slices = list()
    for view in range(len(img.shape)):
        dim_size = img.shape[view]
        non_empty_slices = np.array(
            [sl for sl in range(dim_size) if np.count_nonzero(get_axis(img, view, sl)) > 0])
        num_non_empty = len(non_empty_slices)

        # trying to 5% slices at the tails (bottom clipping at 0)
        skip_count = max(0, np.around(num_non_empty * 0.05).astype('int16'))
        # only when possible
        if skip_count > 0 and (num_non_empty - 2 * skip_count >= num_slices_per_view):
            non_empty_slices = non_empty_slices[skip_count: -skip_count]
            num_non_empty = len(non_empty_slices)

        # sampling non-empty slices only
        sampled_indices = np.linspace(0, num_non_empty, num=min(num_non_empty, num_slices_per_view),
                                      endpoint=False)
        slices_in_dim = non_empty_slices[np.around(sampled_indices).astype('int64')]

        # ensure you do not overshoot
        slices_in_dim = [sn for sn in slices_in_dim if sn >= 0 or sn <= num_non_empty]

        slices.append(slices_in_dim)

    return slices


def scale_images_0to1(slice1, slice2):
    """Scale the two images to [0, 1] based on min/max from both."""

    min_value = max(slice1.min(), slice2.min())
    max_value = max(slice1.max(), slice2.max())

    slice1 = (slice1 - min_value) / max_value
    slice2 = (slice2 - min_value) / max_value

    return slice1, slice2


def verify_sampler(sampler, image, image_shape, view_set, num_slices):
    """verifies the sampler requested is valid."""

    if isinstance(sampler, str):
        sampler = sampler.lower()
        if sampler not in ['linear', ]:
            raise ValueError(f'Sampling strategy: {sampler} not implemented.')
        out_sampler = sampler
        out_sampling_method = 'linear'
    elif isinstance(sampler, Iterable):
        if any([index < 0 or index > 100 for index in sampler]):
            raise ValueError('sampling percentages must be in  [0-100]% range')
        if len(sampler) > min(num_slices):
            num_slices = np.maximum(num_slices, len(sampler))
        out_sampler = np.array(sampler)
        out_sampling_method = 'percentage'
    elif callable(sampler):
        # checking if the callable returns a bool
        for view in view_set:
            middle_slice = int(image_shape[view] / 2)
            if not isinstance(sampler(get_axis(image, view, middle_slice)), bool):
                raise ValueError('sampler callable must return a boolean value (True/False)')

        out_sampler = sampler
        out_sampling_method = 'callable'
    else:
        raise NotImplementedError('Invalid choice for sampler! Choose one of: '
                                  'linear, percentage or callable')

    return out_sampler, out_sampling_method, num_slices


def save_figure(fig, annot=None, output_path=None):

    if annot is not None:
        fig.suptitle(annot, backgroundcolor='black', color='g')

    try:
        fig.tight_layout()
    except:
        pass

    if output_path is not None:
        # output_path = output_path.replace(' ', '_')
        # fig.savefig(output_path + '.png', bbox_inches='tight', dpi=200)
        fig.savefig(output_path, bbox_inches='tight', dpi=200)
