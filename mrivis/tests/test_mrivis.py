
import os

from matplotlib import pyplot as plt
from os.path import join as pjoin, abspath, realpath, basename, dirname, exists as pexists
from mrivis import checkerboard, color_mix, voxelwise_diff
from mrivis.utils import scale_0to1
from pytest import raises

test_dir = dirname(realpath(__file__))
base_dir = realpath(pjoin(test_dir, '..', '..', 'example_datasets'))
out_dir = realpath(pjoin(base_dir, 'test_out'))
os.makedirs(out_dir, exist_ok=True)

highly_mismatched1 = pjoin(base_dir, '3569_bl_PPMI.nii')
highly_mismatched2 = pjoin(base_dir, '4086_bl_PPMI.nii')

slightly_mismatched1 = pjoin(base_dir, '3569_bl_PPMI.nii')
slightly_mismatched2 = pjoin(base_dir, '3569_bl_PPMI_smoothed.nii')

identical1 = pjoin(base_dir, '3569_bl_PPMI.nii')
identical2 = pjoin(base_dir, '3569_bl_PPMI.nii')

im_sets = ((highly_mismatched2, highly_mismatched1, 'different subjects _mismatched_'),
           (slightly_mismatched1, slightly_mismatched2, 'smoothed vs original'),
           (identical1, identical2, 'identical'),
           )

num_rows = 2
num_cols = 5
num_slices = 10

img_lim = None # [0, 4000]
rescaling='each'


def test_voxelwise_diff():

    for im_set in im_sets:
        comb_id = 'voxelwise_diff {}'.format(im_set[2])
        out_path = pjoin(out_dir, comb_id.replace(' ', '_'))
        voxelwise_diff(im_set[0], im_set[1],
                       rescale_method=rescaling,
                       overlay_image=True,
                       overlay_alpha=0.7,
                       background_threshold=25,
                       num_rows=num_rows,
                       num_cols=num_cols,
                       annot=comb_id,
                       output_path=out_path)
        if not pexists(out_path+'.png'):
            raise IOError('expected output file not created:\n'
                          '{}'.format(out_path))


def test_color_mix():

    for im_set in im_sets:
        for alpha in (1.0, ): # np.arange(0.35, 0.95, 0.05):
            comb_id = 'color_mix {} alpha {:0.2f}'.format(im_set[2], alpha)
            out_path = pjoin(out_dir, comb_id.replace(' ', '_'))
            color_mix(im_set[0], im_set[1],
                      alpha_channels=(alpha, alpha),
                      rescale_method=rescaling,
                      num_rows=num_rows,
                      num_cols=num_cols,
                      annot=comb_id,
                      output_path=out_path)
        if not pexists(out_path+'.png'):
            raise IOError('expected output file not created:\n'
                          '{}'.format(out_path))

def test_checkerboard():

    patch_set = (5, 10, 40) # (1, 2, 3, 5, 10, 25, 40)
    for im_set in im_sets:
        for ps in patch_set:
            comb_id = 'checkerboard {} patch size {}'.format(im_set[2], ps)
            out_path = pjoin(out_dir, comb_id.replace(' ', '_'))
            checkerboard(im_set[0], im_set[1],
                         patch_size=ps,
                         rescale_method=rescaling,
                         num_rows=num_rows,
                         num_slices=num_slices,
                         annot=comb_id,
                         output_path=out_path)
            if not pexists(out_path+'.png'):
                raise IOError('expected output file not created:\n'
                              '{}'.format(out_path))

def test_collage():
    from mrivis.base import Collage
    from mrivis.utils import read_image
    img_path = pjoin(base_dir, '3569_bl_PPMI.nii')
    img = read_image(img_path, None)
    scaled = scale_0to1(img)
    c = Collage(num_slices=15, view_set=(0, 1), num_rows=3)
    c.attach(scaled)
    plt.show(block=False)
    print(c)

test_checkerboard()
# test_color_mix()
# test_voxelwise_diff()
# test_collage()
