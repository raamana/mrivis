
import os
import matplotlib
matplotlib.use('Agg')

from matplotlib import pyplot as plt
from os.path import join as pjoin, abspath, realpath, basename, dirname, exists as pexists
from mrivis import checkerboard, color_mix, voxelwise_diff
from mrivis.utils import scale_0to1, read_image
import numpy as np
from mrivis.base import Collage, SlicePicker


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
                      num_slices=num_slices,
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


def test_collage_class():

    img_path = pjoin(base_dir, '3569_bl_PPMI.nii')
    img = read_image(img_path, None)
    scaled = scale_0to1(img)
    c = Collage(num_slices=15, view_set=(0, 1), num_rows=3)

    try:
        c.attach(scaled)
    except:
        raise ValueError('Attach does not work')

    try:
        c.transform_and_attach(scaled, np.square)
    except:
        raise ValueError('transform_and_attach does not work')

    try:
        print(c)
    except:
        raise ValueError('repr implementation failed')


def test_slice_picker():

    img_path = pjoin(base_dir, '3569_bl_PPMI.nii')
    img = read_image(img_path, None)
    sp = SlicePicker(img, num_slices=15, view_set=(0, 1))

    try:
        for dim, sl_num, data in sp.get_slices(extended=True):
            print(dim, sl_num, data.shape)
    except:
        raise ValueError('get_slices() does not work')

    try:
        for d1, d2 in sp.get_slices_multi((img, img)):
            assert np.allclose(d1, d2)
    except:
        raise ValueError('get_slices_multi() does not work')

    try:
        print(sp)
    except:
        raise ValueError('repr implementation failed')


    def density_over(img2d, min_density = 0.65):

        return (np.count_nonzero(img2d.flatten())/img2d.size)<=min_density

    print('testing different sampling strategies .. ')
    for sname, sampler in zip(('linear', 'percent', 'callable'),
                              ('linear', (5, 50, 95), density_over)):
        sp = SlicePicker(img, sampler=sampler)
        print(sname)
        print(repr(sp))

    print('testing linear sampling')
    for ns in np.random.randint(0, min(img.shape), 10):

        sp_linear = SlicePicker(img, sampler='linear', num_slices=ns)
        print(repr(sp_linear))
        if 3*ns != len(sp_linear.get_slice_indices()):
            raise ValueError('error in linear sampling')

    print('testing percentage sampling')
    perc_list = [5, 10, 45, 60, 87]
    sp_perc = SlicePicker(img, sampler=perc_list)
    print(repr(sp_perc))
    if 3*len(perc_list) != len(sp_perc.get_slice_indices()):
        raise ValueError('error in percentage sampling')

    print('testing ability to save to gif')
    import tempfile
    gif_path = tempfile.NamedTemporaryFile(suffix='.gif').name
    print(gif_path)
    sp.save_as_gif(gif_path)
    if not pexists(gif_path):
        raise IOError('Saving to GIF failed')

    try:
        import imageio
        gif = imageio.mimread(gif_path, format='gif')
    except:
        raise ValueError('Saved GIF file could not be read properly!')

    print()

    print()


# test_checkerboard()
# test_color_mix()
# test_voxelwise_diff()
# test_collage_class()
test_slice_picker()
