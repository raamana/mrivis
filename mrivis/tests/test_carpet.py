import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib.axes import Axes
from mrivis.base import Carpet
from os.path import dirname, join as pjoin, realpath, exists as pexists

# test_dir = dirname(realpath(__file__))
# data_dir = realpath(pjoin(test_dir, '..', '..', 'example_datasets'))
# epi_path = pjoin(data_dir, 'example_datasets', 'epi_func.nii')
# img_hdr = nib.load(epi_path)
# img = img_hdr.get_data()

def make_rand_image(img_shape):

    # toy mask
    img = np.zeros(img_shape)
    rand_img = np.random.random(img.shape)

    thresh = 0.8
    rand_mask = rand_img > thresh
    img[rand_mask] = (rand_img[rand_mask]-thresh)/(1-thresh)

    return img

def make_clustered_image(img_shape, num_clusters):

    return np.random.choice(np.arange(num_clusters), img_shape)

min_num_dims = 3
max_num_dims = 4
min_size_in_a_dim = 50
max_size_in_a_dim = 300
max_num_rois = 50

num_dims = np.random.randint(min_num_dims, max_num_dims)
img_shape = tuple(np.random.randint(min_size_in_a_dim, max_size_in_a_dim, num_dims))
num_rois = np.random.randint(1, max_num_rois)

img = make_rand_image(img_shape)
roi = make_clustered_image(img_shape[:-1], num_rois)
carpet = Carpet(img)

print(' num_dims {} \n img shape {}\n num rois {}'.format(num_dims, img_shape, num_rois))

def test_cropped_size():

    if np.count_nonzero(img) != np.count_nonzero(carpet.carpet):
        raise ValueError('cropped image size do not match!')

def test_basic_functionality():

    ax = carpet.show()
    if not isinstance(ax, Axes):
        raise RuntimeError('Carpet.show() method failed')

def test_cluster_data():

    for roi_, roi_type in zip([None, roi], ['None', 'seg']):

        try:
            carpet.cluster_rows_in_roi(roi_)
        except:
            raise RuntimeError('clustering carpet failed - with roi={}'.format(roi_type))
        else:
            try:
                cl_ax = carpet.show(clustered=True)
            except:
                raise RuntimeError('display of clustered carpet failed '
                                   '- with roi={}'.format(roi_type))
            else:
                if not isinstance(cl_ax, Axes):
                    raise RuntimeError('Carpet.show(clustered=True) method failed')

def test_save():

    from tempfile import mktemp
    out_path = mktemp(suffix='.png')
    carpet.show()
    carpet.save(output_path=out_path)
    if not pexists(out_path):
        raise IOError('saving to figure failed!')



test_cluster_data()
