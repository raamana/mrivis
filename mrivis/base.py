import numpy as np
from mrivis.utils import check_views, check_num_slices

class SlicePicker(object):
    """
    Class to pick non-empty slices along the various dimensions for a given image.

    The term `slice` here refers to one cross-section in a 3D image,
        towards which this class is designed for.
        However there are no explicit restrictions placed on dicing N=4+ array
        and receiving a n-1 dim array.
    """

    def __init__(self,
                 image_in,
                 view_set=(0, 1, 2),
                 num_slices=(10, )):
        """
        Constructor.

        Parameters
        ----------
        image_in : ndarray
            3D array to be sliced.
            there are no explicit restrictions placed on number of dimensions for image_in,
             to get a n-1 dim array, but appropriate reshaping may need to be performed.

        view_set : iterable

        num_slices : int or iterable of size as view_set

        """

        self._image = image_in
        self._image_shape = self._image.shape
        self.view_set = check_views(view_set, max_views=len(self._image_shape))
        self.num_slices = check_num_slices(self._image_shape, num_slices)
        self._pick_slices()  # creates self._slices

    def _pick_slices(self):
        """
        Picks the slices to display in each dimension/view,
            skipping any empty slices (without any data at all).
        """

        self._slices = list()
        for view, ns in zip(self.view_set, self.num_slices):
            dim_size = self._image_shape[view]
            non_empty_slices = np.array([sl for sl in range(dim_size) if
                                         np.count_nonzero(self._get_axis(self._image, view, sl)) > 0])
            num_non_empty = len(non_empty_slices)

            # trying to skip 5% slices at the tails (bottom clipping at 0)
            skip_count = max(0, np.around(num_non_empty * 0.05).astype('int16'))
            # only when possible
            if skip_count > 0 and (num_non_empty - 2 * skip_count > ns):
                non_empty_slices = non_empty_slices[skip_count: -skip_count]
                num_non_empty = len(non_empty_slices)

            # sampling non-empty slices only
            sampled_indices = np.linspace(0, num_non_empty, num=min(num_non_empty, ns),
                                          endpoint=False)
            slices_in_dim = non_empty_slices[np.around(sampled_indices).astype('int64')]

            # ensure you do not overshoot
            slices_in_dim = [sn for sn in slices_in_dim if sn >= 0 or sn <= num_non_empty]
            # adding view and slice # at the same time.
            self._slices.extend([(view, slice) for slice in slices_in_dim])

    def _get_axis(self, array, axis, slice_num, extended=False):
        """Returns a fixed axis"""

        slice_list = [slice(None)] * array.ndim
        slice_list[axis] = slice_num
        slice_data = array[slice_list].T  # transpose for proper orientation

        if not extended:
            # return just the slice data
            return slice_data
        else:
            # additionally include which dim and which slice num
            return axis, slice_num, slice_data

    def get_slice_indices(self):
        """Returns indices for the slices selected (each a tuple : (dim, slice_num))"""

        return self._slices

    def get_slices(self, extended=False):
        """Generator over all the slices selected, each time returning a cross-section."""

        for dim, slice_num in self._slices:
            yield self._get_axis(self._image, dim, slice_num, extended=extended)

    def get_slices_multi(self, *image_list, extended=False):
        """Returns the same cross-section from the multiple images supplied.

        All images must be of the same shape as the original image defining this object.
        """

        # ensure all the images have the same shape
        for img in image_list:
            if img.shape != self._image_shape:
                raise ValueError('Supplied images are not compatible with this class. '
                                 'They must have the shape: {}'.format(self._image_shape))

        for dim, slice_num in self._slices:
            multiple_slices = (self._get_axis(img, dim, slice_num) for img in image_list)
            if not extended:
                # return just the slice data
                yield multiple_slices
            else:
                # additionally include which dim and which slice num
                # not using extended option in get_axis, to avoid complicating unpacking
                return dim, slice_num, multiple_slices


    def __iter__(self):
        """Returns the next panel, and the associated dimension and slice number"""

        return iter(self._slices)


if __name__ == '__main__':
    pass
