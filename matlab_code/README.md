
# Matlab scripts to compare quality of registration or spatial similarity (checkerboard plots)

*Note*: this is no longer maintained! The [python version](../../../) is signicantly more powerful, extensible and is highly recommended.

Requires: Freesurfer's Matlab readers (MRIread) available at https://github.com/freesurfer/freesurfer/tree/dev/matlab

## to compare two T1 MRI scans (checkerboard plots)

Usage:

```
mrivis_checkerboard(path_image1, path_image2, patch_size_in_voxels) # square patch
mrivis_checkerboard(image1, image2, patch_size_in_voxels)
mrivis_checkerboard(path_image1, image2, [ width, height ]) # rectangular patch
```

A sample output (when the two scans are mismatched):

![checkerboard](../docs/zoomed_in/vis_all3.png)

When there is mismatch, you can clearly see it (patch size 15 voxels square):

![checkerboard](../docs/zoomed_in/vis_all3_mismatch_ps15.png)

Let's make the patches a bit bigger (patch size 25 voxels square):

![checkerboard](../docs/zoomed_in/vis_all3_mismatch_ps25_axial.png)
![checkerboard](../docs/zoomed_in/vis_all3_mismatch_ps25_sagittal.png)

Let's make the patches a even bigger (50 voxels square):

![checkerboard](../docs/zoomed_in/vis_all3_mismatch_ps50.png)

Let's use a **rectangular** patch (10 voxels high and 30 voxels wide):

![checkerboard](../docs/zoomed_in/vis_all3_mismatch__rect_ps10_30_sagittal.png)
![checkerboard](../docs/zoomed_in/vis_all3_mismatch__rect_ps10_30_axial.png)

If they were identical (no mismatch at all), you won't see any edges or borders:

![identical](../docs/zoomed_in/vis_all3_identical.png)

Full layout with 6x6 pangels can be seen in [this folder](../docs/comprehensive).

## to visualize plans MRI scans (by themselves)

```
mrivis_collage(path_image)
```

![collage](../docs/comprehensive/vis_collage.png)
