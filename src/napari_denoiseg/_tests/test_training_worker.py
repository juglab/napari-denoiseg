import pytest
import numpy as np
from denoiseg.models import DenoiSeg

from napari_denoiseg.utils import generate_config, State, training_worker, get_shape_order
from napari_denoiseg.utils.denoiseg_utils import remove_C_dim
from napari_denoiseg.utils.training_worker import (
    sanity_check_validation_fraction,
    sanity_check_training_size,
    get_validation_patch_shape,
    normalize_images,
    prepare_data_disk,
    load_data_from_disk,
    detect_non_zero_frames,
    create_train_set,
    create_val_set,
    check_napari_data,
    prepare_data_layers
)
from napari_denoiseg._tests.test_utils import (
    create_simple_model,
    create_data
)


@pytest.mark.parametrize('fraction', [0.05, 0.1, 0.5, 1])
def test_sanity_check_high_validation_fraction(fraction):
    n = 5
    m = int(n / fraction)
    X_train = np.zeros((m, 8, 8, 1))
    X_val = np.zeros((n, 8, 8, 1))

    # check validation fraction
    sanity_check_validation_fraction(X_train, X_val)


@pytest.mark.filterwarnings("error")
@pytest.mark.parametrize('fraction', [0.01, 0.02, 0.04, 0.048])
def test_sanity_check_low_validation_fraction(fraction):
    n = 5
    m = int(n / fraction)
    X_train = np.zeros((m, 8, 8, 1))
    X_val = np.zeros((n, 8, 8, 1))

    with pytest.raises(UserWarning):
        sanity_check_validation_fraction(X_train, X_val)


@pytest.mark.parametrize('shape, axes', [((1, 16, 16, 1), 'SXY'),
                                         ((8, 16, 16, 3), 'SXY'),
                                         ((1, 32, 64, 1), 'SXY'),
                                         ((1, 16, 16, 16, 1), 'SZXY'),
                                         ((8, 16, 16, 16, 3), 'SZXY'),
                                         ((1, 32, 128, 64, 1), 'SZXY')])
def test_sanity_check_training_size(tmp_path, shape, axes):
    """
    Test sanity check using acceptable shapes (axes XYZ must be divisible by 16).
    :param tmp_path:
    :param shape:
    :param axes:
    :return:
    """
    model = create_simple_model(tmp_path, shape)

    # run sanity check on training size
    sanity_check_training_size(np.zeros(shape), model, axes)


@pytest.mark.parametrize('shape, axes', [((1, 8, 16, 1), 'SXYC'),
                                         ((1, 16, 66, 1), 'SXYC'),
                                         ((1, 18, 64, 1), 'SXYC'),
                                         ((1, 16, 16, 8, 1), 'SZXYC'),
                                         ((1, 66, 128, 64, 1), 'SZXYC')])
def test_sanity_check_training_size_error(tmp_path, shape, axes):
    """
    Test sanity check using disallowed shapes (axes XYZ must be divisible by 16).
    :param tmp_path:
    :param shape:
    :param axes:
    :return:
    """
    model = create_simple_model(tmp_path, shape)

    # run sanity check on training size
    with pytest.raises(ValueError):
        sanity_check_training_size(np.zeros(shape), model, axes)


@pytest.mark.parametrize('shape_in, shape_out, axes', [((32, 8, 16, 1), (8, 16), 'SXYC'),
                                                       ((64, 8, 16, 32, 1), (16, 32, 8), 'SZXYC')])
def test_get_validation_patch_shape(shape_in, shape_out, axes):
    """
    Test that the validation patch shape returned corresponds to the dimensions ZXY (in order).
    :param shape_in:
    :param shape_out:
    :param axes:
    :return:
    """
    X_val = np.zeros(shape_in)

    assert get_validation_patch_shape(X_val, axes) == shape_out


@pytest.mark.parametrize('shape_train, shape_val, shape_patch',
                         [((100000, 4, 4, 1), (100000, 4, 4, 1), (16, 16)),
                          ((100000, 4, 4, 4, 1), (100000, 4, 4, 4, 1), (16, 16, 16))])
def test_normalize_images(tmp_path, shape_train, shape_val, shape_patch):
    # create data
    np.random.seed(42)
    X_train = np.random.normal(10, 5, shape_train)
    X_val = np.random.normal(10, 5, shape_val)

    # create model
    name = 'myModel'
    config = generate_config(X_train, shape_patch)
    model = DenoiSeg(config, name, tmp_path)

    # normalize data
    X_train_norm, X_val_norm = normalize_images(model, X_train, X_val)
    assert (np.abs(np.mean(X_train_norm, axis=0)) < 0.01).all()
    assert (np.abs(np.std(X_train_norm, axis=0) - 1) < 0.01).all()
    assert (np.abs(np.mean(X_val_norm, axis=0)) < 0.01).all()
    assert (np.abs(np.std(X_val_norm, axis=0) - 1) < 0.01).all()


@pytest.mark.parametrize('shape, axes, final_shape, final_axes',
                         [((16, 8), 'XY', (8, 16), 'YX'),
                          ((16, 12, 8), 'XCY', (8, 16, 12), 'YXC'),
                          ((16, 8, 12), 'XYZ', (12, 8, 16), 'ZYX'),
                          ((16, 3, 8, 5, 10, 12), 'XCYSTZ', (10, 5, 12, 8, 16, 3), 'TSZYXC')])
def test_get_shape_order(shape, axes, final_shape, final_axes):
    ref_axes = 'TSZYXC'
    x = np.zeros(shape)

    new_shape, new_axes, _ = get_shape_order(x.shape, axes, ref_axes)
    assert new_shape == final_shape
    assert new_axes == final_axes


@pytest.mark.parametrize('shape, axes',
                         [((32, 16), 'XY')])
def test_load_data_from_disk_different_XY(tmp_path, shape, axes):
    """
    Load images with different X and Y as numpy array and check that patches are created before augmentation
    (otherwise an error is raised).
    """
    folders = ['train_x', 'train_y']
    sizes = [3, 2]
    shapes = [shape for _ in sizes]

    # create data
    create_data(tmp_path, folders, sizes, shapes)

    # load data
    X, Y_onehot, Y, new_axes = load_data_from_disk(tmp_path / folders[0],
                                                   tmp_path / folders[1],
                                                   axes,
                                                   augmentation=True,
                                                   check_exists=False,
                                                   patch_shape=(4, 4))

    assert X.shape[0] > sizes[0]



@pytest.mark.parametrize('shape, axes, final_shape, final_axes',
                         [((16, 16), 'XY', (16, 16, 1), 'YXC'),
                          ((16, 16, 8), 'XYZ', (8, 16, 16, 1), 'ZYXC'),
                          ((8, 16, 8), 'XZY', (16, 8, 8, 1), 'ZYXC')])
def test_load_data_from_disk_train_XYZ(tmp_path, shape, axes, final_shape, final_axes):
    """
    Load an array of axes XYZ from disk with more training images than labels.
    """
    folders = ['train_x', 'train_y']
    sizes = [3, 2]
    shapes = [shape for _ in sizes]

    # create data
    create_data(tmp_path, folders, sizes, shapes)

    # load data
    X, Y_onehot, Y, new_axes = load_data_from_disk(tmp_path / folders[0],
                                                   tmp_path / folders[1],
                                                   axes,
                                                   augmentation=True,
                                                   check_exists=False)

    assert new_axes == 'S' + final_axes
    assert X.shape == (sizes[0] * 8,) + final_shape
    assert Y_onehot.shape == (sizes[0] * 8,) + final_shape[:-1] + (3,)
    assert Y.shape == (sizes[0] * 8,) + final_shape[:-1]


@pytest.mark.parametrize('shape, axes, final_shape, final_axes',
                         [((16, 16, 8), 'XYT', (8, 16, 16, 1), 'SYXC'),
                          ((16, 16, 12, 8), 'XYTZ', (12, 8, 16, 16, 1), 'SZYXC'),
                          ((8, 12, 16, 8), 'XTZY', (12, 16, 8, 8, 1), 'SZYXC')])
def test_load_data_from_disk_train_T(tmp_path, shape, axes, final_shape, final_axes):
    """
    Load an array with T axis from disk with more training images than labels.
    """
    folders = ['train_x', 'train_y']
    sizes = [3, 2]
    shapes = [shape for _ in sizes]

    # create data
    create_data(tmp_path, folders, sizes, shapes)

    # load data
    X, Y_onehot, Y, new_axes = load_data_from_disk(tmp_path / folders[0],
                                                   tmp_path / folders[1],
                                                   axes,
                                                   augmentation=True,
                                                   check_exists=False)

    assert new_axes == final_axes
    assert X.shape == (sizes[0] * 8 * final_shape[0],) + final_shape[1:]
    assert Y_onehot.shape == (sizes[0] * 8 * final_shape[0],) + final_shape[1:-1] + (3,)
    assert Y.shape == (sizes[0] * 8 * final_shape[0],) + final_shape[1:-1]


@pytest.mark.parametrize('shape, axes, final_shape, final_axes',
                         [((16, 16, 6), 'XYC', (16, 16, 6), 'YXC'),
                          ((16, 12, 16, 8), 'XCYZ', (8, 16, 16, 12), 'ZYXC'),
                          ((5, 8, 16, 8), 'CXZY', (16, 8, 8, 5), 'ZYXC')])
def test_load_data_from_disk_train_C(tmp_path, shape, axes, final_shape, final_axes):
    """
    Load an array with C axis from disk with more training images than labels.
    """
    folders = ['train_x', 'train_y']
    sizes = [3, 2]
    shapes = [shape, remove_C_dim(shape, axes)]  # remove C from y

    # create data
    create_data(tmp_path, folders, sizes, shapes)

    # load data
    X, Y_onehot, Y, new_axes = load_data_from_disk(tmp_path / folders[0],
                                                   tmp_path / folders[1],
                                                   axes,
                                                   augmentation=True,
                                                   check_exists=False)

    assert new_axes == 'S' + final_axes
    assert X.shape == (sizes[0] * 8,) + final_shape
    assert Y_onehot.shape == (sizes[0] * 8,) + final_shape[:-1] + (3,)
    assert Y.shape == (sizes[0] * 8,) + final_shape[:-1]


@pytest.mark.parametrize('shape, axes, final_shape, final_axes',
                         [((16, 16), 'XY', (16, 16, 1), 'YXC'),
                          ((16, 16, 8), 'XYZ', (8, 16, 16, 1), 'ZYXC'),
                          ((8, 16, 8), 'XZY', (16, 8, 8, 1), 'ZYXC')])
def test_load_data_from_list_XYZ(tmp_path, shape, axes, final_shape, final_axes):
    """
    Load an array of axes XYZ as a list with more training images than labels.
    """
    folders = ['train_x', 'train_y']
    sizes = [2, 1]
    shapes = [shape for _ in sizes]

    # create data
    create_data(tmp_path, folders, sizes, shapes)

    # load data
    X, Y_onehot, Y, new_axes = load_data_from_disk(tmp_path / folders[0],
                                                   tmp_path / folders[1],
                                                   axes,
                                                   augmentation=True,
                                                   check_exists=False)

    assert new_axes == 'S' + final_axes
    assert X.shape == (sizes[0] * 8,) + final_shape
    assert Y_onehot.shape == (sizes[0] * 8,) + final_shape[:-1] + (3,)
    assert Y.shape == (sizes[0] * 8,) + final_shape[:-1]


@pytest.mark.parametrize('shape, axes, final_shape, final_axes',
                         [((16, 16, 8), 'XYT', (8, 16, 16, 1), 'SYXC'),
                          ((16, 16, 12, 8), 'XYTZ', (12, 8, 16, 16, 1), 'SZYXC'),
                          ((8, 12, 16, 8), 'XTZY', (12, 16, 8, 8, 1), 'SZYXC')])
def test_load_data_from_list_T(tmp_path, shape, axes, final_shape, final_axes):
    """
    Load an array with T axis as a list with more training images than labels.
    """
    folders = ['train_x', 'train_y']
    sizes = [2, 1]
    shapes = [shape for _ in sizes]

    # create data
    create_data(tmp_path, folders, sizes, shapes)

    # load data
    X, Y_onehot, Y, new_axes = load_data_from_disk(tmp_path / folders[0],
                                                   tmp_path / folders[1],
                                                   axes,
                                                   augmentation=True,
                                                   check_exists=False)

    assert new_axes == final_axes
    assert X.shape == (sizes[0] * 8 * final_shape[0],) + final_shape[1:]
    assert Y_onehot.shape == (sizes[0] * 8 * final_shape[0],) + final_shape[1:-1] + (3,)
    assert Y.shape == (sizes[0] * 8 * final_shape[0],) + final_shape[1:-1]


@pytest.mark.parametrize('shape1, shape2, axes, final_axes',
                         [((16, 16, 3), (8, 8, 3), 'XYC', 'YXC'),
                          ((16, 3, 16, 8), (8, 3, 8, 16), 'XCYZ', 'ZYXC'),
                          ((3, 8, 16, 8), (3, 16, 8, 16), 'CXZY', 'ZYXC')])
def test_load_data_from_list_C(tmp_path, shape1, shape2, axes, final_axes):
    """
    Load an array with C axis as a list with more training images than labels.
    """
    folders = ['train_x', 'train_y']
    sizes = [2, 1]
    shapes1 = [shape1, remove_C_dim(shape1, axes)]  # remove C from y
    shapes2 = [shape2, remove_C_dim(shape2, axes)]  # remove C from y

    # create data
    create_data(tmp_path, folders, sizes, shapes1)
    create_data(tmp_path, folders, sizes, shapes2, prefix='2_')

    if 'Z' in axes:
        patch = (4, 4, 4)
    else:
        patch = (4, 4)

    # load data
    X, Y_onehot, Y, new_axes = load_data_from_disk(tmp_path / folders[0],
                                                   tmp_path / folders[1],
                                                   axes,
                                                   augmentation=True,
                                                   check_exists=False,
                                                   patch_shape=patch)
    # TODO continue


@pytest.mark.parametrize('shape, axes', [((16, 8), 'YX'),
                                         ((16, 16, 17), 'ZYX'),
                                         ((16, 16, 16, 8), 'ZYCX')])
def test_prepare_data_disk_error(tmp_path, shape, axes):
    """
    Test that an error is raised if XY dims are different.
    :param tmp_path:
    :param shape:
    :return:
    """
    folders = ['train_x', 'train_y', 'val_x', 'val_y']
    sizes = [20, 5, 8, 8]
    shapes = [shape, remove_C_dim(shape, axes)]

    # create data
    create_data(tmp_path, folders, sizes, shapes)

    # load data
    with pytest.raises(ValueError):
        prepare_data_disk(tmp_path / folders[0],
                          tmp_path / folders[1],
                          tmp_path / folders[2],
                          tmp_path / folders[3],
                          axes)


@pytest.mark.parametrize('shape, axes', [((16, 16), 'YX'),
                                         ((16, 16, 8), 'XYZ'),
                                         ((16, 16, 8, 3), 'YXZC')])
def test_prepare_data_disk_unpaired_val(tmp_path, shape, axes):
    """
    Test that an error is raised when the number of validation image and labels don't match.
    :param tmp_path:
    :param shape:
    :return:
    """
    folders = ['train_x', 'train_y', 'val_x', 'val_y']
    sizes = [20, 5, 10, 8]
    shapes = [shape, remove_C_dim(shape, axes), shape, remove_C_dim(shape, axes)]

    # create data
    create_data(tmp_path, folders, sizes, shapes)

    # load data
    with pytest.raises(FileNotFoundError):
        prepare_data_disk(tmp_path / folders[0],
                          tmp_path / folders[1],
                          tmp_path / folders[2],
                          tmp_path / folders[3],
                          axes)


@pytest.mark.parametrize('shape', [(10, 16, 8),  # SYX
                                   (10, 12, 8, 16),  # SZYX
                                   (10, 8, 16, 3),  # SYXC
                                   (10, 4, 16, 8, 3),  # SZYXC
                                   ])
def test_detect_non_zero_frames(shape):
    n = 4

    # create images
    one = np.ones(shape[1:])
    zero = np.zeros(shape[1:])

    # random indexing of the zeros
    ind = np.random.choice([i for i in range(shape[0])], shape[0] - n, replace=False)

    # build array
    X = [one for _ in range(shape[0])]
    for i in ind:
        X[i] = zero

    im = np.stack(X)
    assert im.shape == shape
    assert len(detect_non_zero_frames(im)) == n


def test_detect_non_zero_frames_edge_cases():
    assert len(detect_non_zero_frames(np.zeros((1, 8, 8)))) == 1
    assert len(detect_non_zero_frames(np.ones((1, 8, 8)))) == 0


@pytest.mark.parametrize('shape, axes', [((20, 16, 16, 1), 'SYXC'),
                                         ((20, 12, 16, 16, 1), 'SZYXC'),
                                         ((20, 16, 16, 3), 'SYXC'),
                                         ((20, 4, 16, 16, 3), 'SZYXC')])
def test_create_train_set(shape, axes):
    n = 6  # number of labeled frames
    m = 3  # size difference between x and y along the S dimension

    # random indexing for the split
    ind = np.random.choice([i for i in range(shape[0] - m)], shape[0] - n, replace=False)

    # create x and y
    template = np.ones(shape[1:])
    x = np.stack([i * template for i in range(shape[0])])
    y = np.stack([i * template[..., 0] for i in range(shape[0] - m)])
    assert x.shape[1:-1] == y.shape[1:] == shape[1:-1]

    # split
    X, Y = create_train_set(x, y, ind, axes)
    assert Y.shape[0] == X.shape[0] == (shape[0]-len(ind)) * 8  # augmentation and added missing labels
    assert X.shape == (n * 8,) + shape[1:]
    assert Y.shape == (n * 8,) + shape[1:-1]

    # check that the right frames were selected
    ind_x = tuple([0 for i in range(1, len(shape))])
    vals_x = X[:, ind_x]
    for v in vals_x:
        assert v not in ind

    ind_y = tuple([0 for i in range(1, len(shape) - 1)])
    vals_y = Y[:, ind_y]
    for v in vals_y:
        assert v not in ind


@pytest.mark.parametrize('shape', [(20, 16, 16, 1),
                                   (20, 12, 16, 16, 1),
                                   (20, 16, 16, 3),
                                   (20, 4, 16, 16, 3)])
def test_create_val_set(shape):
    n = 4  # number of labeled frames
    m = 3  # size difference between x and y along the S dimension

    # random indexing for the split
    ind = np.random.choice([i for i in range(shape[0] - m)], n, replace=False)

    # create x and y
    template = np.ones(shape[1:])
    x = np.stack([i * template for i in range(shape[0])])
    y = np.stack([i * template[..., 0] for i in range(shape[0] - m)])
    assert x.shape[1:-1] == y.shape[1:] == shape[1:-1]

    # split
    X, Y = create_val_set(x, y, ind)
    assert Y.shape[0] == X.shape[0] == len(ind)
    assert X.shape == (n,) + shape[1:]
    assert Y.shape == (n,) + shape[1:-1]


@pytest.mark.parametrize('shape, axes', [((20, 16, 16), 'TXY'),  # No S
                                         ((20, 16, 16), 'XYS'),  # wrong YX order
                                         ((20, 16, 16), 'SXY'),  # wrong YX order
                                         ((16, 16), 'YX'),  # No 3rd dim
                                         ((20, 16, 16), 'SZYX'),  # axes and data not matching
                                         ((20, 16, 18), 'SZYX'),  # YX not square
                                         ((20, 3, 16, 16), 'SCYX')])  # Labels with C dim
def test_check_napari_data_errors(shape, axes):
    x = np.random.random(shape)
    y = np.random.random(shape)

    with pytest.raises(ValueError):
        check_napari_data(x, y, axes)


def test_check_napari_data_xy_dims():
    x = np.random.random((5, 16, 16))
    y = np.random.random((5, 16, 16, 5))

    with pytest.raises(ValueError):
        check_napari_data(x, y, 'SYX')


def test_check_napari_data_different_xy():
    x = np.random.random((5, 16, 16))
    y = np.random.random((5, 17, 17))

    with pytest.raises(ValueError):
        check_napari_data(x, y, 'SYX')


@pytest.mark.parametrize('shape1, shape2, axes',
                         [((5, 3, 16, 16), (5, 2, 16, 16), 'SCYX'),
                          ((5, 3, 16, 16), (5, 4, 16, 16), 'SCYX'),
                          ((5, 3, 16, 16), (5, 3, 16, 16), 'SCYX'),
                          ((3, 5, 16, 16), (2, 5, 16, 16), 'CSYX'),
                          ((3, 5, 16, 16), (4, 5, 16, 16), 'CSYX'),
                          ((5, 20, 3, 16, 16), (5, 20, 2, 16, 16), 'TSCYX')])
def test_check_napari_data_C_error(shape1, shape2, axes):
    x = np.random.random(shape1)
    y = np.random.random(shape2)

    with pytest.raises(ValueError):
        check_napari_data(x, y, axes)


@pytest.mark.parametrize('shape1, shape2, axes',
                         [((5, 3, 16, 16), (5, 16, 16), 'SCYX'),
                          ((5, 3, 16, 16), (5, 16, 16), 'SCYX'),
                          ((5, 3, 16, 16), (5, 16, 16), 'SCYX'),
                          ((3, 5, 16, 16), (5, 16, 16), 'CSYX'),
                          ((3, 5, 16, 16), (5, 16, 16), 'CSYX'),
                          ((5, 20, 3, 16, 16), (5, 20, 16, 16), 'TSCYX')])
def test_check_napari_data_C(shape1, shape2, axes):
    x = np.random.random(shape1)
    y = np.random.random(shape2)

    check_napari_data(x, y, axes)


@pytest.mark.qt
def test_prepare_data_layers_no_labels(make_napari_viewer):
    shape = (10, 8, 8)

    # create data
    x = np.random.random(shape)
    y = np.zeros(shape, dtype=np.uint16)

    # make viewer and add layers
    viewer = make_napari_viewer()

    viewer.add_image(x, name='X')
    viewer.add_labels(y, name='Y')

    # check layers
    assert viewer.layers['X'].data.shape == shape
    assert viewer.layers['Y'].data.shape == shape

    # raise error because there is too little data
    with pytest.raises(ValueError):
        prepare_data_layers(viewer.layers['X'].data,
                            viewer.layers['Y'].data,
                            50,
                            'SYX')


@pytest.mark.qt
@pytest.mark.parametrize('perc', [0, 10, 22, 93, 95, 100])
def test_prepare_data_layers_not_enough_data(make_napari_viewer, perc):
    shape = (50, 8, 8)
    n_unlabeled = shape[0] - 20  # 20 labeled frames

    # create data
    x = np.random.random(shape)
    y = np.random.randint(0, 255, shape, dtype=np.uint16)

    # choose empty frames
    y[:n_unlabeled, ...] = np.zeros((n_unlabeled, shape[1], shape[2]), dtype=np.uint16)

    # make viewer and add layers
    viewer = make_napari_viewer()

    viewer.add_image(x, name='X')
    viewer.add_labels(y, name='Y')

    # check layers
    assert viewer.layers['X'].data.shape == shape
    assert viewer.layers['Y'].data.shape == shape

    # raise error because there is too little data
    with pytest.raises(ValueError):
        prepare_data_layers(viewer.layers['X'].data,
                            viewer.layers['Y'].data,
                            perc,
                            'SYX')


@pytest.mark.qt
@pytest.mark.parametrize('shape, axes, final_axes',
                         [((20, 8, 8), 'SYX', 'SYXC'),
                          ((20, 2, 8, 8), 'SZYX', 'SZYXC')])
def test_prepare_data_layers_no_CT(make_napari_viewer, shape, axes, final_axes):
    n_unlabeled = shape[axes.find('S')] - 10
    perc = 70

    # create data
    x = np.random.random(shape)

    # Y can have smaller S size
    y = np.random.randint(0, 255, (shape[0]-2, *shape[1:]), dtype=np.uint16)
    y[:n_unlabeled, ...] = np.zeros((n_unlabeled, *shape[1:]), dtype=np.uint16)

    # make viewer and add layers
    viewer = make_napari_viewer()

    viewer.add_image(x, name='X')
    viewer.add_labels(y, name='Y')

    # check layers
    assert viewer.layers['X'].data.shape == shape
    assert viewer.layers['Y'].data.shape == (shape[0]-2, *shape[1:])

    # prepare data
    X, Y, X_val, Y_val, y_val_no_hot, new_axes = prepare_data_layers(viewer.layers['X'].data,
                                                                     viewer.layers['Y'].data,
                                                                     perc,
                                                                     axes)
    assert X.shape[0]/8 + X_val.shape[0] == shape[0]
    assert X.shape[1:] == X_val.shape[1:]

    assert Y.shape[0]/8 + Y_val.shape[0] == shape[0]
    assert Y.shape[1:-1] == Y_val.shape[1:-1] == y_val_no_hot.shape[1:]
    assert Y.shape[-1] == Y_val.shape[-1] == 3

    assert new_axes == final_axes


@pytest.mark.qt
def test_prepare_data_layers_ZSYX(make_napari_viewer):
    axes = 'ZSYX'
    shape_x = (2, 20, 8, 8)
    shape_y = (2, 18, 8, 8)  # Y can have smaller dim along S
    n_unlabeled = shape_x[axes.find('S')] - 10
    shape_zeros = (2, n_unlabeled, 8, 8)
    perc = 70

    # create data
    x = np.random.random(shape_x)

    # Y can have smaller S size
    y = np.random.randint(0, 255, shape_y, dtype=np.uint16)
    y[:, :n_unlabeled, ...] = np.zeros(shape_zeros, dtype=np.uint16)

    # make viewer and add layers
    viewer = make_napari_viewer()

    viewer.add_image(x, name='X')
    viewer.add_labels(y, name='Y')

    # check layers
    assert viewer.layers['X'].data.shape == shape_x
    assert viewer.layers['Y'].data.shape == shape_y

    # prepare data
    X, Y, X_val, Y_val, y_val_no_hot, new_axes = prepare_data_layers(viewer.layers['X'].data,
                                                                     viewer.layers['Y'].data,
                                                                     perc,
                                                                     axes)
    assert X.shape[0]/8 + X_val.shape[0] == shape_x[1]
    assert X.shape[1:] == X_val.shape[1:]

    assert Y.shape[0]/8 + Y_val.shape[0] == shape_x[1]
    assert Y.shape[1:-1] == Y_val.shape[1:-1] == y_val_no_hot.shape[1:]
    assert Y.shape[-1] == Y_val.shape[-1] == 3

    assert new_axes == 'SZYXC'


@pytest.mark.qt
@pytest.mark.parametrize('shape, axes, final_axes',
                         [((20, 4, 8, 8), 'STYX', 'SYXC'),
                          ((4, 20, 8, 8), 'TSYX', 'SYXC'),
                          ((20, 5, 4, 8, 8), 'SZTYX', 'SZYXC'),
                          ((20, 4, 5, 8, 8), 'STZYX', 'SZYXC'),
                          ((4, 20, 5, 8, 8), 'TSZYX', 'SZYXC')])
def test_prepare_data_layers_T_no_C(make_napari_viewer, shape, axes, final_axes):
    n_unlabeled = shape[axes.find('S')] - 10
    perc = 70

    # create data
    x = np.random.random(shape)

    # Y can have smaller S size
    shape_x = shape
    shape_y = list(shape)
    shape_y[axes.find('S')] = shape_x[axes.find('S')] - 2

    y = np.random.randint(0, 255, shape_y, dtype=np.uint16)

    ind_S = axes.find('S')
    ind_T = axes.find('T')
    if ind_S != 0:
        # TODO there must be an elegant way to access subarray with dynamical indices...
        shape_zero = shape_y.copy()
        shape_zero[ind_S] = n_unlabeled
        zeros = np.zeros(shape_zero, dtype=np.uint16)

        if ind_S == 1:
            y[:, :n_unlabeled, ...] = zeros
        else:
            y[:, :, :n_unlabeled, ...] = zeros

    else:
        y[:n_unlabeled, ...] = np.zeros((n_unlabeled, *shape_y[1:]), dtype=np.uint16)

    # make viewer and add layers
    viewer = make_napari_viewer()

    viewer.add_image(x, name='X')
    viewer.add_labels(y, name='Y')

    # prepare data
    X, Y, X_val, Y_val, y_val_no_hot, new_axes = prepare_data_layers(viewer.layers['X'].data,
                                                                     viewer.layers['Y'].data,
                                                                     perc,
                                                                     axes)
    assert X.shape[0]/8 + X_val.shape[0] == shape[ind_S] * shape[ind_T]
    assert X.shape[1:] == X_val.shape[1:]

    assert Y.shape[0]/8 + Y_val.shape[0] == shape[ind_S] * shape[ind_T]
    assert Y.shape[1:-1] == Y_val.shape[1:-1] == y_val_no_hot.shape[1:]
    assert Y.shape[-1] == Y_val.shape[-1] == 3

    assert new_axes == final_axes


@pytest.mark.qt
@pytest.mark.parametrize('shape, axes, final_axes',
                         [((20, 4, 3, 8, 8), 'STCYX', 'SYXC'),
                          ((3, 4, 20, 8, 8), 'CTSYX', 'SYXC'),
                          ((20, 3, 5, 4, 8, 8), 'SCZTYX', 'SZYXC'),
                          ((20, 4, 5, 3, 8, 8), 'STZCYX', 'SZYXC'),
                          ((4, 20, 5, 3, 8, 8), 'TSCZYX', 'SZYXC')])
def test_prepare_data_layers_CT(make_napari_viewer, shape, axes, final_axes):
    n_unlabeled = shape[axes.find('S')] - 10
    perc = 70

    # create data
    x = np.random.random(shape)

    # find axes
    ind_S = axes.find('S')
    ind_T = axes.find('T')
    ind_C = axes.find('C')

    # Y can have smaller S size
    shape_y = list(shape)
    shape_y[ind_S] = shape_y[ind_S] - 2
    shape_y = remove_C_dim(shape_y, axes)

    if ind_S > ind_C != -1:
        ind_S_y = ind_S - 1
    else:
        ind_S_y = ind_S

    y = np.random.randint(0, 255, shape_y, dtype=np.uint16)

    if ind_S_y != 0:
        # TODO there must be an elegant way to access subarray with dynamical indices...
        shape_zero = list(shape_y)
        shape_zero[ind_S_y] = n_unlabeled
        zeros = np.zeros(shape_zero, dtype=np.uint16)

        if ind_S_y == 1:
            y[:, :n_unlabeled, ...] = zeros
        else:
            y[:, :, :n_unlabeled, ...] = zeros

    else:
        y[:n_unlabeled, ...] = np.zeros((n_unlabeled, *shape_y[1:]), dtype=np.uint16)

    # make viewer and add layers
    viewer = make_napari_viewer()

    viewer.add_image(x, name='X')
    viewer.add_labels(y, name='Y')

    # prepare data
    X, Y, X_val, Y_val, y_val_no_hot, new_axes = prepare_data_layers(viewer.layers['X'].data,
                                                                     viewer.layers['Y'].data,
                                                                     perc,
                                                                     axes)
    assert X.shape[0]/8 + X_val.shape[0] == shape[ind_S] * shape[ind_T]
    assert X.shape[1:] == X_val.shape[1:]
    assert X.shape[-1] == X_val.shape[-1] == shape[ind_C]

    assert Y.shape[0]/8 + Y_val.shape[0] == shape[ind_S] * shape[ind_T]
    assert Y.shape[1:-1] == Y_val.shape[1:-1] == y_val_no_hot.shape[1:]
    assert Y.shape[-1] == Y_val.shape[-1] == 3

    assert new_axes == final_axes


@pytest.mark.qt
@pytest.mark.timeout(10)
@pytest.mark.parametrize('shape1, shape2, axes',
                         [((10, 16, 16), (10, 16, 16), 'SYX'),
                          ((2, 5, 16, 16), (2, 5, 16, 16), 'STYX'),
                          ((10, 16, 16, 16), (10, 16, 16, 16), 'SZYX'),
                          ((10, 3, 16, 16), (10, 16, 16), 'SCYX'),
                          ((16, 2, 5, 16, 16), (16, 2, 5, 16, 16), 'ZTSYX')])
def test_train_napari(qtbot, make_napari_viewer, tmp_path, shape1, shape2, axes):

    class Value:
        def __init__(self, value):
            self.value = value

        def get_value(self):
            return self.value()

    class Slider:
        def __init__(self, func):
            self.slider = Value(func)

    class MonkeyPatchWidget:
        def __init__(self, napari_viewer):
            from napari_denoiseg.widgets import AxesWidget

            self.images = Value(napari_viewer.layers['X'])
            self.labels = Value(napari_viewer.layers['Y'])
            self.perc_train_slider = Slider(lambda: 80)
            self.axes = axes
            self.n_epochs = 1
            self.n_steps = 1
            self.batch_size_spin = Value(lambda: 2)
            self.patch_size_XY = Value(lambda: 16)
            self.patch_size_Z = Value(lambda: 16)
            self.is_3D = 'Z' in axes
            self.tf_version = ''
            self.state = State.RUNNING
            self.model = None
            self.threshold = 0.1
            self.inputs = None
            self.outputs = None
            self.new_axes = None
            self.load_from_disk = False
            self.axes_widget = AxesWidget(n_axes=len(shape1), is_3D='Z' in axes)
            self.axes_widget.set_text_field(axes)

    # make viewer and add layers
    viewer = make_napari_viewer()

    # create data
    x = np.ones(shape1)
    y = np.ones(shape2, dtype=np.uint16)

    # add layers
    viewer.add_image(x, name='X')
    viewer.add_labels(y, name='Y')

    # start training
    widget = MonkeyPatchWidget(viewer)
    t = training_worker(widget)

    with qtbot.waitSignal(t.finished, timeout=50_000):
        t.start()

    assert isinstance(widget.model, DenoiSeg)

# TODO: test continue training with other weights
