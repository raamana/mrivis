import numpy as np
from mrivis.utils import check_views, check_num_slices, get_axis

class SlicePicker(object):
    """
    Class to pick non-empty slices along the various dimensions for a given image.
    """

    def __init__(self,
                 image_in,
                 view_set=(0, 1, 2),
                 num_slices=(10, )):

        self._image = image_in
        self._image_shape = self._image.shape
        self.view_set = check_views(view_set, max_views=len(self._image_shape))
        self.num_slices = check_num_slices(self._image_shape, num_slices)
        self._pick_slices()  # creates self.slices

    def _pick_slices(self):
        """
        Picks the slices to display in each dimension/view,
            skipping any empty slices (without any data at all).
        """

        self._slices = list()
        for view, ns in zip(self.view_set, self.num_slices):
            dim_size = self._image_shape[view]
            non_empty_slices = np.array([sl for sl in range(dim_size) if
                                         np.count_nonzero(get_axis(self._image, view, sl)) > 0])
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

    def get_slice_indices(self):
        """Returns the indices for the slices selected (each a tuple : (dim, slice_num))"""

        return self._slices

    def get_slices(self):
        """Generator over all the slices selected, each time returning 2D image"""

        for dim, slice_num in self._slices:
            yield get_axis(self._image, dim, slice_num)

    def get_slices_multi(self, *image_list):
        """Returns the same slice from the multiple images supplied."""

        # ensure all the images have the same shape
        for img in image_list:
            if img.shape != self._image_shape:
                raise ValueError('Supplied images are not compatible with this class. '
                                 'They must have the shape: {}'.format(self._image_shape))

        for dim, slice_num in self._slices:
            yield (get_axis(img, dim, slice_num) for img in image_list)

    def __iter__(self):
        """Returns the next panel, and the associated dimension and slice number"""

        return iter(self._slices)


if __name__ == '__main__':
    pass
