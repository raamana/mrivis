from genericpath import exists as pexists
from os.path import realpath

import nibabel as nib
import numpy as np
from matplotlib import pyplot as plt
import matplotlib as mpl

def _diff_image(slice1, slice2,
                abs_value=True,
                cmap='gray',
                **kwargs):
    """Computes the difference image"""

    diff = slice1-slice2

    if abs_value:
        diff = np.abs(diff)

    return diff, cmap


def diff_colormap():
    "Custom colormap to map low values to black or another color."

    # bottom = plt.cm.copper(np.linspace(0., 1, 6))
    black  = np.atleast_2d([0., 0., 0., 1.])
    bottom = np.repeat(black, 6, axis=0)
    middle = plt.cm.copper(np.linspace(0, 1, 250))
    # remain = plt.cm.Reds(np.linspace(0, 1, 240))

    colors = np.vstack((bottom, middle))
    diff_colormap = mpl.colors.LinearSegmentedColormap.from_list('diff_colormap', colors)

    return diff_colormap


def get_axis(array, axis, slice_num):
    """Returns a fixed axis"""

    slice_list = [slice(None)] * array.ndim
    slice_list[axis] = slice_num
    slice_data = array[slice_list].T  # transpose for proper orientation

    return slice_data


def check_int(num, num_descr, min_value=0):
    """Validation and typecasting."""

    if not np.isfinite(num) or num < min_value:
        raise ValueError('{} is not finite or is not >= {}'.format(num_descr, min_value))

    return int(num)


def check_patch_size(patch_size):
    """Validation and typcasting"""

    patch_size = np.array(patch_size)
    if patch_size.size == 1:
        patch_size = np.repeat(patch_size, 2).astype('int16')

    return patch_size


def check_params(num_rows, num_cols, padding):
    """Validation and typcasting"""

    num_rows = check_int(num_rows, 'num_rows', min_value=1)
    num_cols = check_int(num_cols, 'num_cols', min_value=1)
    padding = check_int(padding, 'padding', min_value=0)

    return num_rows, num_cols, padding


def read_image(img_spec, bkground_thresh):
    """Image reader. Removes stray values close to zero (smaller than 5 %ile)."""

    if isinstance(img_spec, str):
        if pexists(realpath(img_spec)):
            hdr = nib.load(img_spec)
            # trying to stick to an orientation
            hdr = nib.as_closest_canonical(hdr)
            img = hdr.get_data()
        else:
            raise IOError('Given path to image does not exist!')
    elif isinstance(img_spec, np.ndarray):
        img = img_spec
    else:
        raise ValueError('Invalid input specified! '
                         'Input either a path to image data, or provide 3d Matrix directly.')

    img = check_image_is_3d(img)

    if not np.issubdtype(img.dtype, np.float):
        img = img.astype('float32')

    return threshold_image(img, bkground_thresh)


def check_image_is_3d(img):
    """Ensures the image loaded is 3d and nothing else."""

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


def threshold_image(img, bkground_thresh, bkground_value=0.0):
    """
    Thresholds a given image at a value or percentile.

    Replacement value can be specified too.
    """

    if bkground_thresh is None:
        return img

    if isinstance(bkground_thresh, str):
        try:
            thresh_perc = float(bkground_thresh.replace('%', ''))
        except:
            raise ValueError('percentile specified could not be parsed correctly - must be a string of the form "5%", "10%" etc')
        else:
            thresh_value = np.percentile(img, thresh_perc)
    elif isinstance(bkground_thresh, (float, int)):
        thresh_value = bkground_thresh
    else:
        raise ValueError('Invalid specification for background threshold.')

    img[img < thresh_value] = bkground_value

    return img


def scale_0to1(image):
    """Scale the two images to [0, 1] based on min/max from both."""

    min_value = image.min()
    max_value = image.max()
    image = (image - min_value) / (max_value-min_value)

    return image


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
    "Crops an image or slice to its extents"

    if padding < 1:
        return img

    beg_coords, end_coords = crop_coords(img, padding)

    if len(img.shape)==3:
        img = crop_3dimage(img, beg_coords, end_coords)
    elif len(img.shape)==2:
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
        non_empty_slices = np.array([sl for sl in range(dim_size) if np.count_nonzero(get_axis(img, view, sl)) > 0])
        num_non_empty = len(non_empty_slices)

        # trying to 5% slices at the tails (bottom clipping at 0)
        skip_count = max(0, np.around(num_non_empty*0.05).astype('int16'))
        # only when possible
        if skip_count > 0 and (num_non_empty-2*skip_count>=num_slices_per_view):
            non_empty_slices = non_empty_slices[skip_count : -skip_count]
            num_non_empty = len(non_empty_slices)

        # sampling non-empty slices only
        sampled_indices = np.linspace(0, num_non_empty, num=min(num_non_empty, num_slices_per_view), endpoint=False)
        slices_in_dim = non_empty_slices[ np.around(sampled_indices).astype('int64') ]

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
