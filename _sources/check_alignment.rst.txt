---------------
Alignment check
---------------

Usage of the different methods are shown below:


The purpose of ``checkerboard``, ``voxelwise_diff`` and ``color_mix`` functions in ``mrivis`` is to offer different ways of checking alignment between two two given images (either from the same modality or different modalities possibly with different contrast properties).

To use them is simply a matter of importing them e.g. ``checkerboard``, and passing the two images to compare:

.. code-block:: python

    from mrivis import checkerboard

    path1 = '/Users/Reddy/Desktop/image.nii'
    path2 = '/Users/Reddy/Desktop/another.nii'

    checkerboard(path1, path2) # square patches


You could customize them further in various ways using different parameters to fit your needs:

.. code-block:: python

    checkerboard(path1, path2, patch_size=5)

    checkerboard(path1, path2, rescale_method=(0, 256) )

    checkerboard(path1, path2, patch_size=10,
                 num_rows=1, num_cols=3) # 1 row per dimension, with 3 columns

    checkerboard(path1, path2, patch_size=[10, 20], # rectangular patches
                 num_rows=2, # 2 rows per dimension (6 rows in total)
                 num_cols=5) # 5 panels per row


You can use the other methods ``voxelwise_diff`` and ``color_mix`` in a very similar manner:

.. code-block:: python

    from mrivis import voxelwise_diff, color_mix

    voxelwise_diff(path1, path2)
    voxelwise_diff(path1, path2, abs_value=False)

    color_mix(path1, path2, alpha_channels=(1, 1))
    color_mix(path1, path2, alpha_channels=(0.7, 1))



Gallery of examples to check alignment
---------------------------------------

Download this image to get a better look at the differences:

.. figure:: flyer_option_matrix.png


Comparing two BOLD images
-------------------------

.. figure:: flyer_haxby_mean_BOLD_subj_1_2.png
   :alt: flyer\_haxby\_mean\_BOLD\_subj\_1\_2


Sample outputs for checkerboard plots
-------------------------------------

When the two scans are mismatched:

.. figure:: zoomed_in/vis_all3.png
   :alt: vis\_all3

When the mismatch is low (here a smoothed image is comapred to its
original), you can see the differences in intensity (due to smoothing),
but you can also see that they are both spatially aligned accurately:

.. figure:: flyer2_low_mismatch.png
   :alt: flyer2\_low\_mismatch

With really low patch-sizes (e.g. 1, which is voxel-wise alternation),
you can see the alignment even better:

.. figure:: zoomed_in/vis_voxelwise_axial.png
   :alt: vis\_voxelwise\_axial


When there is mismatch, you can clearly see it (patch size 15 voxels
square):

.. figure:: zoomed_in/vis_all3_mismatch_ps15.png
   :alt: vis\_all3\_mismatch\_ps15


Let's make the patches a bit bigger (patch size 25 voxels square):

.. figure:: zoomed_in/vis_all3_mismatch_ps25_axial.png

Let's make the patches a even bigger (50 voxels square):

.. figure:: zoomed_in/vis_all3_mismatch_ps50.png
   :alt: vis\_all3\_mismatch\_ps50


Let's use a **rectangular** patch (10 voxels high and 30 voxels wide):

.. figure:: zoomed_in/vis_all3_mismatch__rect_ps10_30_sagittal.png
.. figure:: zoomed_in/vis_all3_mismatch__rect_ps10_30_axial.png

If they were identical (no mismatch at all), you won't see any edges or
borders:

.. figure:: zoomed_in/vis_all3_identical.png
   :alt: identical

   identical

Full layout with 6x6 pangels can be seen in `this
folder <https://github.com/raamana/mrivis/tree/master/docs/comprehensive>`__.
