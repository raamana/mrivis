
# mrivis

`mrivis` offers
- tools for comparison of spatial similarity of 3d MRI scans (T1, T2, PET etc) using checkerboard plots and other comparison methods (see below).
- classes to develop customizable collages of 3d images - [demo](https://nbviewer.jupyter.org/github/raamana/mrivis/blob/master/docs/example_notebooks/mrivis_demo_Collage_SlicePicker_classes.ipynb)


![](https://img.shields.io/pypi/v/mrivis.svg)


## Installation

```bash
pip install -U mrivis
```

## Usage:

Note: this package is in ACTIVE development.
Expect minor bugs, along frequent fixes/changes.


```python
from mrivis import checkerboard, voxelwise_diff, color_mix

path1 = '/Users/Reddy/Desktop/image.nii'
path2 = '/Users/Reddy/Desktop/another.nii'

checkerboard(path1, path2, patch_size=5) # square patches

checkerboard(path1, path2, rescale_method=(0, 256) )

checkerboard(path1, path2, patch_size=10,
             num_rows=1, num_cols=3) # 1 row per dimension, with 3 columns

checkerboard(path1, path2, patch_size=[10, 20], # rectangular patches
             num_rows=2, # 2 rows per dimension (6 rows in total)
             num_cols=5) # 5 panels per row

voxelwise_diff(path1, path2)
voxelwise_diff(path1, path2, abs_value=False)

color_mix(path1, path2, alpha_channels=(1, 1))
color_mix(path1, path2, alpha_channels=(0.7, 1))

```

## Various options available

Download this image to get a better look at the differences:

![flyer](docs/flyer_option_matrix.png)


## Comparing two BOLD images

![flyer_haxby_mean_BOLD_subj_1_2](docs/flyer_haxby_mean_BOLD_subj_1_2.png)

## Sample outputs for checkerboard plots

When the two scans are mismatched:

![vis_all3](docs/zoomed_in/vis_all3.png)

When the mismatch is low (here a smoothed image is comapred to its original),
you can see the differences in intensity (due to smoothing),
but you can also see that they are both spatially aligned accurately:

![flyer2_low_mismatch](docs/flyer2_low_mismatch.png)

With really low patch-sizes (e.g. 1, which is voxel-wise alternation), you can see the alignment even better:

![vis_voxelwise_axial](docs/zoomed_in/vis_voxelwise_axial.png)

When there is mismatch, you can clearly see it (patch size 15 voxels square):

![vis_all3_mismatch_ps15](docs/zoomed_in/vis_all3_mismatch_ps15.png)

Let's make the patches a bit bigger (patch size 25 voxels square):

![vis_all3_mismatch_ps25_axial](docs/zoomed_in/vis_all3_mismatch_ps25_axial.png)
![vis_all3_mismatch_ps25_sagittal](docs/zoomed_in/vis_all3_mismatch_ps25_sagittal.png)

Let's make the patches a even bigger (50 voxels square):

![vis_all3_mismatch_ps50](docs/zoomed_in/vis_all3_mismatch_ps50.png)

Let's use a **rectangular** patch (10 voxels high and 30 voxels wide):

![vis_all3_mismatch__rect_ps10_30_sagittal](docs/zoomed_in/vis_all3_mismatch__rect_ps10_30_sagittal.png)
![vis_all3_mismatch__rect_ps10_30_axial](docs/zoomed_in/vis_all3_mismatch__rect_ps10_30_axial.png)

If they were identical (no mismatch at all), you won't see any edges or borders:

![identical](docs/zoomed_in/vis_all3_identical.png)

Full layout with 6x6 pangels can be seen in [this folder](docs/comprehensive).

## More example outputs

Many more example outputs, for different combinations of parameters, can be seen in this [example outputs folder in docs](docs/example_outputs).






