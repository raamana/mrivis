
# mrivis

Tools for comparison of registration and spatial similarity of 3d MRI scans (T1, T2, PET etc) using checkerboard plots.


![https://img.shields.io/pypi/v/mrivis.svg](https://pypi.python.org/pypi/mrivis)


## to compare two T1 MRI scans (checkerboard plots)

Usage:

```bash
mrivis --checkerboard path_image1 path_image2 --patch_size 10 # square patch

mrivis --checkerboard path_image1 path_image2 --patch_size 10 30 # rectangular patch

```

A sample output (when the two scans are mismatched):

![vis_all3](../docs/zoomed_in/vis_all3.png)

When there is mismatch, you can clearly see it (patch size 15 voxels square):

![vis_all3_mismatch_ps15](../docs/zoomed_in/vis_all3_mismatch_ps15.png)

Let's make the patches a bit bigger (patch size 25 voxels square):

![vis_all3_mismatch_ps25_axial](../docs/zoomed_in/vis_all3_mismatch_ps25_axial.png)
![vis_all3_mismatch_ps25_sagittal](../docs/zoomed_in/vis_all3_mismatch_ps25_sagittal.png)

Let's make the patches a even bigger (50 voxels square):

![vis_all3_mismatch_ps50](../docs/zoomed_in/vis_all3_mismatch_ps50.png)

Let's use a **rectangular** patch (10 voxels high and 30 voxels wide):

![vis_all3_mismatch__rect_ps10_30_sagittal](../docs/zoomed_in/vis_all3_mismatch__rect_ps10_30_sagittal.png)
![vis_all3_mismatch__rect_ps10_30_axial](../docs/zoomed_in/vis_all3_mismatch__rect_ps10_30_axial.png)

If they were identical (no mismatch at all), you won't see any edges or borders:

![identical](../docs/zoomed_in/vis_all3_identical.png)

Full layout with 6x6 pangels can be seen in [this folder](../docs/comprehensive).

## to visualize plans MRI scans (by themselves) using a collage

```bash
mrivis --collage path_image1 [path_image2]

```

![collage](../docs/comprehensive/vis_collage.png)


