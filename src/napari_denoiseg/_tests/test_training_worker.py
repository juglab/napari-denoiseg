import pytest
import numpy as np
from denoiseg.models import DenoiSeg

from napari_denoiseg.utils import generate_config
from napari_denoiseg.utils.training_worker import (
    sanity_check_validation_fraction,
    sanity_check_training_size,
    get_validation_patch_shape,
    normalize_images,
    reshape_data,
    augment_data,
    prepare_data_disk,
    load_data_from_disk,
    zero_sum,
    list_diff,
    create_train_set,
    create_val_set,
    prepare_data_layers
)
from napari_denoiseg._tests.test_utils import (
    create_model,
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
    model = create_model(tmp_path, shape)

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
    model = create_model(tmp_path, shape)

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
                         [((16, 8), 'YX', (1, 16, 8), 'SYX'),
                          ((16, 8), 'XY', (1, 8, 16), 'SYX'),
                          ((16, 3, 8), 'XZY', (1, 3, 8, 16), 'SZYX'),
                          ((16, 3, 12), 'SXY', (16, 12, 3), 'SYX'),
                          ((5, 5, 1), 'XYZ', (1, 1, 5, 5), 'SZYX'),
                          ((16, 3, 12, 8), 'XCYS', (8, 12, 16, 3), 'SYXC'),
                          ((16, 3, 12, 8), 'ZXCY', (1, 16, 8, 3, 12), 'SZYXC'),
                          ((16, 3, 12, 8), 'XCYZ', (1, 8, 12, 16, 3), 'SZYXC'),
                          ((16, 3, 12, 8), 'ZYXC', (1, 16, 3, 12, 8), 'SZYXC'),
                          ((16, 3, 21, 12, 8), 'ZYSXC', (21, 16, 3, 12, 8), 'SZYXC'),
                          ((16, 3, 21, 12, 8), 'SZYXC', (16, 3, 21, 12, 8), 'SZYXC'),
                          ((16, 3, 8), 'YXT', (8, 16, 3), 'SYX'),
                          ((16, 3, 8), 'XTY', (3, 8, 16), 'SYX'),
                          ((16, 3, 8, 5, 12), 'STZYX', (16 * 3, 8, 5, 12), 'SZYX'),
                          ((16, 3, 8, 5, 12, 6), 'XCSYTZ', (12 * 8, 6, 5, 16, 3), 'SZYXC')])
def test_reshape_data(shape, axes, final_shape, final_axes):
    x = np.zeros(shape)
    y = np.zeros(shape)

    _x, _y, new_axes = reshape_data(x, y, axes)

    assert _x.shape == final_shape
    assert _y.shape == final_shape
    assert new_axes == final_axes


def test_augment_data_simple():
    axes = 'SYX'
    r = np.array([
        [[1, 2], [3, 4]],
        [[5, 6], [7, 8]],
    ])
    r1 = np.array([
        [[2, 4], [1, 3]],
        [[6, 8], [5, 7]],
    ])
    r2 = np.array([
        [[4, 3], [2, 1]],
        [[8, 7], [6, 5]],
    ])
    r3 = np.array([
        [[3, 1], [4, 2]],
        [[7, 5], [8, 6]],
    ])
    f0 = np.array([
        [[3, 4], [1, 2]],
        [[7, 8], [5, 6]],
    ])
    f1 = np.array([
        [[1, 3], [2, 4]],
        [[5, 7], [6, 8]],
    ])
    f2 = np.array([
        [[2, 1], [4, 3]],
        [[6, 5], [8, 7]],
    ])
    f3 = np.array([
        [[4, 2], [3, 1]],
        [[8, 6], [7, 5]],
    ])
    x_final = np.concatenate([r, r1, r2, r3, f0, f1, f2, f3], axis=0)

    x_aug = augment_data(r, axes)
    assert x_aug.shape == x_final.shape
    assert (x_aug == x_final).all()


@pytest.mark.parametrize('shape, axes', [((1, 16, 16), 'SYX'),
                                         ((8, 16, 16), 'SYX'),
                                         ((1, 10, 16, 16), 'SZYX'),
                                         ((32, 10, 16, 16), 'SZYX'),
                                         ((1, 10, 16, 16, 3), 'SZYXC'),
                                         ((32, 10, 16, 16, 3), 'SZYXC')])
def test_augment_data(shape, axes):
    x = np.random.randint(0, 65535, shape, dtype=np.uint16)
    x_aug = augment_data(x, axes)

    assert x_aug.shape == (x.shape[0]*8,) + x.shape[1:]


@pytest.mark.parametrize('shape, axes, final_shape, final_axes',
                         [((16, 16), 'XY', (16, 16), 'YX'),
                          ((16, 16, 8), 'XYZ', (8, 16, 16), 'ZYX'),
                          ((8, 8, 16, 32), 'XYZC', (16, 8, 8, 32), 'ZYXC'),
                          ((12, 8, 16, 8), 'CYZX', (16, 8, 8, 12), 'ZYXC')])
def test_load_data_from_disk_train(tmp_path, shape, axes, final_shape, final_axes):
    folders = ['train_x', 'train_y']
    sizes = [20, 5]

    # create data
    create_data(tmp_path, folders, sizes, shape)

    # load data
    X, Y, x, y, new_axes = load_data_from_disk(tmp_path / folders[0],
                                               tmp_path / folders[1],
                                               axes,
                                               augmentation=True,
                                               check_exists=False)

    assert new_axes == 'S' + final_axes

    assert X.shape[0] == sizes[0] * 8  # augmentation
    assert Y.shape[0] == sizes[0] * 8
    assert x.shape[0] == sizes[0] * 8
    assert y.shape[0] == sizes[0] * 8
    if 'C' in axes:
        assert X.shape == (sizes[0] * 8,) + final_shape
        assert x.shape == (sizes[0] * 8,) + final_shape
        assert Y.shape == (sizes[0] * 8,) + final_shape[:-1] + (3,)
        assert y.shape == (sizes[0] * 8,) + final_shape
    else:
        assert X.shape == (sizes[0] * 8,) + final_shape + (1,)
        assert x.shape == (sizes[0] * 8,) + final_shape + (1,)
        assert Y.shape == (sizes[0] * 8,) + final_shape + (3,)
        assert y.shape == (sizes[0] * 8,) + final_shape + (1,)


@pytest.mark.parametrize('shape', [(16, 8), (32, 16, 16), (32, 16, 16, 8)])
def test_prepare_data_disk_error(tmp_path, shape):
    """
    Test that an error is raised if XY dims are different.
    :param tmp_path:
    :param shape:
    :return:
    """
    folders = ['train_x', 'train_y', 'val_x', 'val_y']
    sizes = [20, 5, 8, 8]

    # create data
    create_data(tmp_path, folders, sizes, shape)

    # load data
    with pytest.raises(ValueError):
        prepare_data_disk(tmp_path / folders[0],
                          tmp_path / folders[1],
                          tmp_path / folders[2],
                          tmp_path / folders[3])


@pytest.mark.parametrize('shape', [(16, 16), (16, 16, 8), (16, 16, 8, 3)])
def test_prepare_data_disk_unpaired_val(tmp_path, shape):
    """
    Test that an error is raised when the number of validation image and labels don't match.
    :param tmp_path:
    :param shape:
    :return:
    """
    folders = ['train_x', 'train_y', 'val_x', 'val_y']
    sizes = [20, 5, 10, 8]

    # create data
    create_data(tmp_path, folders, sizes, shape)

    # load data
    with pytest.raises(FileNotFoundError):
        prepare_data_disk(tmp_path / folders[0],
                          tmp_path / folders[1],
                          tmp_path / folders[2],
                          tmp_path / folders[3])


@pytest.mark.parametrize('shape', [(8,), (8, 8, 16, 16, 32)])
def test_prepare_data_disk_wrong_dims(tmp_path, shape):
    """
    Test that
    :param tmp_path:
    :param shape:
    :return:
    """
    folders = ['train_x', 'train_y', 'val_x', 'val_y']
    sizes = [20, 5, 8, 8]

    # create data
    create_data(tmp_path, folders, sizes, shape)

    # load data
    with pytest.raises(ValueError):
        prepare_data_disk(tmp_path / folders[0],
                          tmp_path / folders[1],
                          tmp_path / folders[2],
                          tmp_path / folders[3])


@pytest.mark.parametrize('shape', [(8,), (16, 16), (16, 16, 8), (32, 16, 16, 8), (32, 16, 16, 8, 3)])
def test_zero_sum(shape):
    pass


# TODO make tests to verify that we deal with multiple dimension
# TODO array mismatch if X!=Y, make test for that
@pytest.mark.parametrize('shape', [(16, 16), (16, 16, 8)])
@pytest.mark.parametrize('perc', [0, 10, 20, 50, 60, 80, 100])
def test_prepare_data_layers(make_napari_viewer, shape, perc):
    sizes = [20, 10]
    shape_X = (sizes[0],) + shape
    shape_Y = (sizes[1],) + shape

    # make viewer and add layers
    viewer = make_napari_viewer()

    viewer.add_image(np.random.random(shape_X), name='X')
    viewer.add_labels(np.random.randint(0, 255, shape_Y, dtype=np.uint16), name='Y')

    # prepare data
    assert viewer.layers['X'].data.shape == shape_X
    assert viewer.layers['Y'].data.shape == shape_Y
    prepare_data_layers(viewer.layers['X'].data, viewer.layers['Y'].data, perc)
