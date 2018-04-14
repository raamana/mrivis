__all__ = ['SlicePicker', 'Collage']

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
from mpl_toolkits.axes_grid1 import ImageGrid
from mrivis.utils import check_int, check_num_slices, check_views


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
        self._image_shape = np.array(self._image.shape)
        self.view_set = check_views(view_set, max_views=len(self._image_shape))
        self.num_slices = check_num_slices(num_slices,
                                           img_shape=self._image_shape[self.view_set],
                                           num_dims=len(self.view_set))
        self._pick_slices()  # creates self._slices

    def _pick_slices(self):
        """
        Picks the slices to display in each dimension/view,
            skipping any empty slices (without any data at all).
        """

        not_empty = lambda vu, sl: np.count_nonzero(
            self._get_axis(self._image, vu, sl)) > 0

        self._slices = list()
        for view, ns in zip(self.view_set, self.num_slices):
            dim_size = self._image_shape[view]
            non_empty_slices = np.array([sl for sl in range(dim_size)
                                         if not_empty(view, sl)])
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
            if img.shape != self._image.shape:
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

    def __len__(self):
        """Returns the total number of slices across all the views."""

        return len(self._slices)

    def __str__(self):

        return 'views : {}\n#slices: {}'.format(self.view_set, self.num_slices)

    def __repr__(self):

        sv = [ [] for _ in self.view_set]
        for v, d in self._slices:
            sv[v].append(d)

        dim_repr = list()
        for v in self.view_set:
            dim_repr.append('{} slices in dim {} : {}'.format(len(sv[v]), v, sv[v]))

        return '\n'.join(dim_repr)



class Collage(object):
    """
    Class exhibiting multiple slices from a 3D image,
        with convenience routines handling all the cross-sections as a single set.
    """


    def __init__(self,
                 view_set=(0, 1, 2),
                 num_rows=2,
                 num_slices=(12,),
                 display_params=None,
                 fig=None,
                 figsize=(14, 11),
                 bounding_rect=(0.02, 0.02, 0.98, 0.98),
                 ):
        """Constructor."""

        self.view_set = check_views(view_set, max_views=3)
        self.num_slices = check_num_slices(num_slices, img_shape=None,
                                           num_dims=len(self.view_set))
        self._make_layout(fig, figsize, num_rows, bounding_rect=bounding_rect)

        if display_params is None:
            self.display_params = dict(interpolation='none', origin='lower',
                                       aspect='equal', cmap='gray', vmin=0.0, vmax=1.0)
        else:
            self.display_params = display_params

        self._data_attached = False


    def _make_layout(self,
                     fig,
                     figsize=(14, 10),
                     num_rows_per_view=2,
                     bounding_rect=(0.03, 0.93, 0.97, 0.97),
                     grid_pad=0.01,
                     grid_aspect=False):

        plt.style.use('dark_background')
        if fig is None:
            self.fig = plt.figure(figsize=figsize)
        else:
            self.fig = fig

        total_num_rows = len(self.view_set) * num_rows_per_view
        total_num_panels = sum(self.num_slices)
        num_cols_per_row = int(np.ceil(total_num_panels / total_num_rows))

        left, bottom, width, height = bounding_rect
        avail_height = height - bottom
        num_views = len(self.view_set)
        height_each_view = (avail_height - num_views * grid_pad) / num_views
        effective_height_each_view = height_each_view + grid_pad

        self.grids = list()
        for ix, view in enumerate(self.view_set):
            rect = (left, bottom + ix * effective_height_each_view,
                    width, height_each_view)
            ig = ImageGrid(self.fig, rect=rect,
                           nrows_ncols=(num_rows_per_view, num_cols_per_row),
                           axes_pad=0.005, aspect=grid_aspect,
                           share_all=True, direction='row')
            self.grids.append(ig)
            # self._set_aspect_ratio(view, ig)

        # flattened for easy access
        self.flat_grid = [ax for gg in self.grids for ax in gg]
        self._set_axes_off()


    def _set_aspect_ratio(self, view, ig):
        """Sets the default properties for each axes"""

        # pick sizes in the remaining dimensions
        panel_sizes = [size for ix, size in enumerate(self.image.shape) if ix != view]
        # compute ratio
        aspect_ratio = panel_sizes[0] / panel_sizes[1]
        # apply it to each of the axes
        for ax in ig:
            ax.set(aspect=aspect_ratio)


    def _set_axes_off(self):
        """Turns off all the x and y axes in each Axis"""

        for ax in self.flat_grid:
            ax.axis('off')


    def show(self, grid=None):
        """Makes the collage visible."""

        self._set_visible(True, grid_index=grid)


    def attach(self, image_in, show=True):
        """Attaches the relevant cross-sections to each axis"""

        self.slicer = SlicePicker(image_in=image_in,
                                  view_set=self.view_set,
                                  num_slices=self.num_slices)

        try:
            for ax, slice_data in zip(self.flat_grid, self.slicer.get_slices()):
                ax.imshow(slice_data, **self.display_params)
        except:
            raise ValueError('unable to attach the given image data to current collage')
        else:
            self._data_attached = True

        # show all the axes
        if show:
            self.show()


    def hide(self, grid=None):
        """Removes the collage from view."""

        self._set_visible(False, grid_index=grid)


    def _set_visible(self, visibility, grid_index=None):
        """Sets the visibility property of all axes."""

        if grid_index is None:
            for ax in self.flat_grid:
                ax.set_visible(visibility)
        else:
            for ax in self.grids[grid_index]:
                ax.set_visible(visibility)


if __name__ == '__main__':
    pass
