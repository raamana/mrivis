__all__ = ['SlicePicker', 'Collage']

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.image import AxesImage
from collections import Iterable

from mrivis.utils import check_num_slices, check_views
from mrivis import config as cfg

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
                 view_set=cfg.view_set_default,
                 num_slices=cfg.num_slices_default,
                 sampler=cfg.sampler_default,
                 min_density=cfg.min_density_default):
        """
        Class to pick non-empty slices along the various dimensions for a given image.

        Parameters
        ----------
        image_in : ndarray
            3D array to be sliced.
            there are no explicit restrictions placed on number of dimensions for image_in,
             to get a n-1 dim array, but appropriate reshaping may need to be performed.

        view_set : iterable
            List of integers selecting the dimesnions to be sliced.

        num_slices : int or iterable of size as view_set
            Number of slices to be selected in each view.

        sampler : str or list or callable
            selection strategy: to identify the type of sampling done to select the slices to return.
            All sampling is done between the first and last non-empty slice in that view/dimension.

            - if 'linear' : linearly spaced slices
            - if list, it is treated as set of percentages at which slices to be sampled
                (must be in the range of [1-100], not [0-1]).
                This could be used to more/all slices in the middle e.g. range(40, 60, 5)
                    or at the end e.g. [ 5, 10, 15, 85, 90, 95]
            - if callable, it must take a 2D image of arbitray size, return True/False
                to indicate whether to select that slice or not.
                Only non-empty slices (atleas one non-zero voxel) are provided as input.
                Simple examples for callable could be based on
                1) percentage of non-zero voxels > x etc
                2) presence of desired texture ?
                3) certain properties of distribution (skewe: dark/bright, energy etc) etc

                If the sampler returns more than requested `num_slices`,
                    only the first num_slices will be selected.

        min_density : float or None
            mininum density of non-zero voxels within a given slice to consider it non-empty
            Default: 0.01 (1%).
            if None, include all slices.

        """

        if len(image_in.shape) < 3:
            raise ValueError('Image must be atleast 3D')
        else:
            self._image = image_in

        self._check_min_density(min_density)

        self._image_shape = np.array(self._image.shape)
        self.view_set = check_views(view_set, max_views=len(self._image_shape))
        self.num_slices = check_num_slices(num_slices,
                                           img_shape=self._image_shape[self.view_set],
                                           num_dims=len(self.view_set))

        self._verify_sampler(sampler)

        self._pick_slices()  # creates self._slices

    def _verify_sampler(self, sampler):

        if isinstance(sampler, str):
            sampler = sampler.lower()
            if sampler not in ['linear', ]:
                raise ValueError('Sampling strategy: {} not implemented.'.format(sampler))
            self._sampler = sampler
            self._sampling_method = 'linear'
        elif isinstance(sampler, Iterable):
            if any([index < 0 or index > 100 for index in sampler]):
                raise ValueError('sampling percentages must be in  [0-100]% range')
            if len(sampler) > min(self.num_slices):
                self.num_slices = np.maximum(self.num_slices, len(sampler))
            self._sampler = np.array(sampler)
            self._sampling_method = 'percentage'
        elif callable(sampler):
            # checking if the callable returns a bool
            for view in self.view_set:
                middle_slice = int(self._image_shape[view] / 2)
                if not isinstance(sampler(self._get_axis(self._image, view, middle_slice)), bool):
                    raise ValueError('sampler callable must return a boolean value (True/False)')

            self._sampler = sampler
            self._sampling_method = 'callable'
        else:
            raise NotImplementedError('Invalid choice for sampler! Choose one of: '
                                      'linear, percentage or callable')

    def _check_min_density(self, min_density):
        """Validator to ensure proper usage."""

        if min_density is None:
            self._min_density = -np.Inf
        elif ( isinstance(min_density, float) and (min_density>=0.0 and min_density<1.0)):
            self._min_density = min_density
        else:
            raise ValueError('min_density must be float and be >=0.0 and < 1.0')

    def _pick_slices(self):
        """
        Picks the slices to display in each dimension/view,
            skipping any empty slices (without any data at all).
        """

        self._slices = list()
        self._slices_by_dim = list()
        for view, ns_in_view in zip(self.view_set, self.num_slices):
            # discarding completely empty or almost empty slices.
            dim_size = self._image_shape[view]
            non_empty_slices = np.array([sl for sl in range(dim_size) if self._not_empty(view, sl)])

            # sampling according to the chosen strategy
            slices_dim = self._sample_slices_in_dim(view, ns_in_view, non_empty_slices)
            #  the following loop is needed to preserve order, while eliminating duplicates
            # list comprehension over a set(slices_dim) wouldn't preserve order
            uniq_slices = list()
            for sn in slices_dim:
                if sn >= 0 and sn < dim_size and sn not in uniq_slices:
                    uniq_slices.append(sn)

            self._slices_by_dim.append(slices_dim)
            # adding view and slice # at the same time..
            self._slices.extend([(view, sn) for sn in slices_dim])

    def _not_empty(self, view, slice_):
        """Checks if the density is too low. """

        img2d = self._get_axis(self._image, view, slice_)
        return (np.count_nonzero(img2d) / img2d.size) > self._min_density

    def _sample_slices_in_dim(self, view, num_slices, non_empty_slices):
        """Samples the slices in the given dimension according the chosen strategy."""

        if self._sampling_method == 'linear':
            return self._linear_selection(non_empty_slices=non_empty_slices, num_slices=num_slices)
        elif self._sampling_method == 'percentage':
            return self._percent_selection(non_empty_slices=non_empty_slices)
        elif self._sampling_method == 'callable':
            return self._selection_by_callable(view=view, non_empty_slices=non_empty_slices,
                                               num_slices=num_slices)
        else:
            raise NotImplementedError('Invalid state for the class!')

    def _linear_selection(self, non_empty_slices, num_slices):
        """Selects linearly spaced slices in given"""

        num_non_empty = len(non_empty_slices)

        # # trying to skip 5% slices at the tails (bottom clipping at 0)
        # skip_count = max(0, np.around(num_non_empty * 0.05).astype('int16'))
        # # only when possible
        # if skip_count > 0 and (num_non_empty - 2 * skip_count > num_slices):
        #     non_empty_slices = non_empty_slices[skip_count: -skip_count]
        #     num_non_empty = len(non_empty_slices)

        sampled_indices = np.linspace(0, num_non_empty, num=min(num_non_empty, num_slices),
                                      endpoint=False)
        slices_in_dim = non_empty_slices[np.around(sampled_indices).astype('int64')]

        return slices_in_dim

    def _percent_selection(self, non_empty_slices):
        """Chooses slices at a given percentage between the first and last non-empty slice."""

        return np.around(self._sampler * len(non_empty_slices) / 100).astype('int64')

    def _selection_by_callable(self, view, num_slices, non_empty_slices):
        """Returns all the slices selected by the given callable."""

        selected = [sl for sl in non_empty_slices
                    if self._sampler(self._get_axis(self._image, view, sl))]

        return selected[:num_slices]

    def _get_axis(self, array, axis, slice_num,
                  extended=False,
                  transpose=True):
        """Returns a fixed axis"""

        slice_list = [slice(None)] * array.ndim
        slice_list[axis] = slice_num
        if transpose:
            # transpose for proper orientation
            slice_data = array[slice_list].T

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

    def get_slices_multi(self, image_list, extended=False):
        """Returns the same cross-section from the multiple images supplied.

        All images must be of the same shape as the original image defining this object.

        image_list : Iterable
            containing atleast 2 images
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
                yield dim, slice_num, multiple_slices

    def __iter__(self):
        """Returns the next panel, and the associated dimension and slice number"""

        return iter(self._slices)

    def __len__(self):
        """Returns the total number of slices across all the views."""

        return len(self._slices)

    def __format__(self, format_spec='s'):
        """various formats"""

        if format_spec in ['s', 'simple']:
            return self.__str__()
        elif format_spec in ['f', 'full']:
            return self.__repr__()
        else:
            return 'invalid format requsted!!'

    def __str__(self):

        return 'views : {}\n' \
               '#slices: {}\n' \
               'sampler: {}'.format(self.view_set, self.num_slices, self._sampling_method)

    def __repr__(self):

        dim_repr = list()
        for ix, vw in enumerate(self.view_set):
            dim_repr.append('{} slices in dim {} : '
                            '{}'.format(len(self._slices_by_dim[ix]),
                                        vw, self._slices_by_dim[ix]))
        return '\n'.join(dim_repr)


class MiddleSlicePicker(SlicePicker):
    """Convenience class to select the classic one middle slice from all views."""

    def __init__(self, image):
        """Returns the middle slice from all views in the image."""

        super().__init__(image_in=image,
                         view_set=cfg.view_set_default,
                         num_slices=1,
                         sampler=(50,), )


class Collage(object):
    """
    Class exhibiting multiple slices from a 3D image,
        with convenience routines handling all the cross-sections as a single set.
    """

    def __init__(self,
                 view_set=cfg.view_set_default,
                 num_rows=cfg.num_rows_per_view_default,
                 num_slices=cfg.num_slices_default,
                 sampler=cfg.sampler_default,
                 attach_image=None,
                 bounding_rect=cfg.bounding_rect_default,
                 fig=None,
                 figsize=cfg.figsize_default,
                 display_params=None,
                 ):
        """
        Class exhibiting multiple slices from a 3D image,
        with convenience routines handling all the cross-sections as a single set.

        Once created with certain `display_params` (containing vmin and vmax),
            this class does NOT automatically rescale the data, as you attach different images.
            Ensure the input images are rescaled to [0, 1] BEFORE attaching.

        Parameters
        ----------

        view_set : iterable
            List of integers selecting the dimesnions to be sliced.

        num_slices : int or iterable of size as view_set
            Number of slices to be selected in each view.

        num_rows : int
            Number of rows per view.

        sampler : str or list or callable
            selection strategy to identify the type of sampling done
            to select the slices to return. All sampling is done between
            the first and last non-empty slice in that view/dimension.

            - if 'linear' : linearly spaced slices
            - if list, it is treated as set of percentages at which slices to be sampled
                (must be in the range of [1-100], not [0-1]).
                This could be used to more/all slices in the middle e.g. range(40, 60, 5)
                    or at the end e.g. [ 5, 10, 15, 85, 90, 95]
            - if callable, it must take a 2D image of arbitray size, return True/False
                to indicate whether to select that slice or not.
                Only non-empty slices (atleas one non-zero voxel) are provided as input.
                Simple examples for callable could be based on
                1) percentage of non-zero voxels > x etc
                2) presence of desired texture ?
                3) certain properties of distribution (skewe: dark/bright, energy etc) etc

                If the sampler returns more than requested `num_slices`,
                    only the first num_slices will be selected.

        attach_image : ndarray
            The image to be attached to the collage, once it is created.
            Must be atleast 3d.

        display_params : dict
            dict of keyword parameters that can be passed to matplotlib's `Axes.imshow()`

        fig : matplotlib.Figure
            figure handle to create the collage in.
            If not specified, creates a new figure.

        figsize : tuple of 2
            Figure size (width, height) in inches.

        bounding_rect : tuple of 4
            The rectangular area to bind the collage to (in normalized figure coordinates)

        """

        self.view_set = check_views(view_set, max_views=3)
        self.num_slices = check_num_slices(num_slices, img_shape=None,
                                           num_dims=len(self.view_set))
        # TODO find a way to validate the input-- using utits.verify_sampler commonly?
        self.sampler = sampler

        if display_params is None:
            self.display_params = dict(interpolation='none', origin='lower',
                                       aspect='equal', cmap='gray', vmin=0.0, vmax=1.0)
        else:
            self.display_params = display_params
        self._make_layout(fig, figsize, num_rows, bounding_rect=bounding_rect)
        if attach_image is not None:
            self.attach(attach_image)
        else:
            self._data_attached = False

    def _make_layout(self,
                     fig,
                     figsize=cfg.figsize_default,
                     num_rows_per_view=cfg.num_rows_per_view_default,
                     bounding_rect=cfg.bounding_rect_default,
                     grid_pad=cfg.grid_pad_default,
                     axis_pad=cfg.axis_pad_default,
                     **axis_kwargs):

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
            ax_grid = self._make_grid_of_axes(bounding_rect=rect, axis_pad=axis_pad,
                                              num_rows=num_rows_per_view,
                                              num_cols=num_cols_per_row,
                                              **axis_kwargs)
            self.grids.append(ax_grid)

        # flattened for easy access
        self.flat_grid = [ax for gg in self.grids for ax in gg]
        # create self.images with one image in each axis
        self._create_imshow_objects()

    def _make_grid_of_axes(self,
                           bounding_rect=cfg.bounding_rect_default,
                           num_rows=cfg.num_rows_per_view_default,
                           num_cols=cfg.num_cols_grid_default,
                           axis_pad=cfg.axis_pad_default,
                           commn_annot=None,
                           **axis_kwargs):
        """Creates a grid of axes bounded within a given rectangle."""

        axes_in_grid = list()
        extents = self._compute_cell_extents_grid(bounding_rect=bounding_rect,
                                                  num_cols=num_cols,
                                                  num_rows=num_rows, axis_pad=axis_pad)
        for cell_ext in extents:
            ax_cell = self.fig.add_axes(cell_ext, frameon=False, visible=False,
                                        **axis_kwargs)
            if commn_annot is not None:
                ax_cell.set_title(commn_annot)
            ax_cell.set_axis_off()
            axes_in_grid.append(ax_cell)

        return axes_in_grid

    @staticmethod
    def _compute_cell_extents_grid(bounding_rect=(0.03, 0.03, 0.97, 0.97),
                                   num_rows=2, num_cols=6,
                                   axis_pad=0.01):
        """
        Produces array of num_rows*num_cols elements each containing the rectangular extents of
        the corresponding cell the grid, whose position is within bounding_rect.
        """

        left, bottom, width, height = bounding_rect
        height_padding = axis_pad * (num_rows + 1)
        width_padding = axis_pad * (num_cols + 1)
        cell_height = float((height - height_padding) / num_rows)
        cell_width = float((width - width_padding) / num_cols)

        cell_height_padded = cell_height + axis_pad
        cell_width_padded = cell_width + axis_pad

        extents = list()
        for row in range(num_rows - 1, -1, -1):
            for col in range(num_cols):
                extents.append((left + col * cell_width_padded,
                                bottom + row * cell_height_padded,
                                cell_width, cell_height))

        return extents

    def _create_imshow_objects(self):
        """Turns off all the x and y axes in each Axis"""

        # uniform values for initial image can cause weird behaviour with normalization
        #       as imshow.set_data() does not automatically update the normalization!!
        # using random data is a better choice
        random_image = np.random.rand(20, 20)
        self.images = [None] * len(self.flat_grid)
        for ix, ax in enumerate(self.flat_grid):
            self.images[ix] = ax.imshow(random_image, **self.display_params)

    def show(self, grid=None):
        """Makes the collage visible."""

        self._set_visible(True, grid_index=grid)

    def attach(self,
               image_in,
               sampler=None,
               show=True):
        """Attaches the relevant cross-sections to each axis"""

        if len(image_in.shape) < 3:
            raise ValueError('Image must be atleast 3D')

        # allowing the choice of new sampling for different invocations.
        if sampler is None:
            temp_sampler = self.sampler
        else:
            temp_sampler = sampler

        slicer = SlicePicker(image_in=image_in,
                             view_set=self.view_set,
                             num_slices=self.num_slices,
                             sampler=temp_sampler)

        try:
            for img_obj, slice_data in zip(self.images, slicer.get_slices()):
                img_obj.set_data(slice_data)
        except:
            self._data_attached = False
            raise ValueError('unable to attach the given image data to current collage')
        else:
            self._data_attached = True

        # show all the axes
        if show:
            self.show()

    def transform_and_attach(self,
                             image_list,
                             func,
                             show=True):
        """
        Displays the transformed (combined) version of the cross-sections from each image,
            (same slice and dimension). So if you input n>=1 images, n slices are obtained
            from each image, which are passed to the func (callable) provided, and the
            result will be displayed in the corresponding cell of the collage.
            Useful applications:
            - input two images, a function to overlay edges of one image on the other
            - input two images, a function to mix them in a checkerboard pattern
            - input one image, a function to saturate the upper half of intensities
                (to increase contrast and reveal any subtle ghosting in slices)

        func must be able to receive as many arguments as many elements in image_list.
            if your func needs additional parameters, make them keyword arguments, and
            use functools.partial to obtain a new callable that takes in just the slices.

        Parameters
        -----------
        image_list : list or ndarray
            list of images or a single ndarray

        func : callable
            function to be applied on the input images (their slices)
                to produce a single slice to be displayed.

        show : bool
            flag to indicate whether make the collage visible.

        """

        if not callable(func):
            raise TypeError('func must be callable!')

        if not isinstance(image_list, (tuple, list)) and isinstance(image_list, np.ndarray):
            image_list = [image_list, ]

        if len(image_list) > 1:
            shape1 = image_list[0].shape
            for ii in range(1, len(image_list)):
                if image_list[ii].shape != shape1:
                    raise ValueError('All images must be of same shape!')
                if len(image_list[ii].shape) < 3:
                    raise ValueError('All images must be atleast 3D')

        slicer = SlicePicker(image_in=image_list[0],
                             view_set=self.view_set,
                             num_slices=self.num_slices)

        try:
            for img_obj, slice_list in zip(self.images,
                                           slicer.get_slices_multi(image_list)):
                img_obj.set_data(func(*slice_list))
        except:
            self._data_attached = False
            raise ValueError('unable to attach mix of given images to current collage')
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
            if grid_index < 0 or grid_index >= len(self.grids):
                raise IndexError('Valid indices : 0 to {}'.format(len(self.grids) - 1))
            for ax in self.grids[grid_index]:
                ax.set_visible(visibility)

    def save(self, annot=None, output_path=None):
        """Saves the collage to disk as an image."""

        if annot is not None:
            self.fig.suptitle(annot, backgroundcolor='black', color='g')

        if output_path is not None:
            output_path = output_path.replace(' ', '_')
            # TODO improve bbox calculations to include ONLY the axes from collage
            # and nothing else
            self.fig.savefig(output_path + '.png', bbox_inches='tight', dpi=200,
                             bbox_extra_artists=self.flat_grid)

    def clear(self):
        """Clears all the axes to start fresh."""

        for ax in self.flat_grid:
            for im_h in ax.findobj(AxesImage):
                im_h.remove()


class MidCollage(Collage):
    """Convenience class to display the mid-slices from all the views."""

    def __init__(self,
                 image,
                 bounding_rect=cfg.bounding_rect_default,
                 fig=None,
                 display_params=None,
                 ):
        """Display mid-slices from all the views.

        image : ndarray
            The image to be attached to the collage, once it is created.
            Must be atleast 3d.

        fig : matplotlib.Figure
            figure handle to create the collage in.
            If not specified, creates a new figure.

        bounding_rect : tuple of 4
            The rectangular area to bind the collage to (in normalized figure coordinates)

        display_params : dict
            dict of keyword parameters that can be passed to matplotlib's `Axes.imshow()`

        """

        super().__init__(view_set=cfg.view_set_default,
                         num_rows=1, num_slices=1, sampler=(50, ),
                         attach_image=image,
                         fig=fig, bounding_rect=bounding_rect,
                         display_params=display_params)



if __name__ == '__main__':
    pass
