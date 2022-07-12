import os
from pathlib import Path
import numpy as np
import pytest

from marshmallow import ValidationError
from napari_denoiseg._tests.test_utils import (
    save_img,
    create_data,
    create_model,
    save_weights_h5,
    create_model_zoo_parameters
)
from napari_denoiseg.utils import (
    load_pairs_generator,
    generate_config,
    build_modelzoo,
    load_weights,
    load_from_disk,
    load_pairs_from_disk,
    remove_C_dim,
    filter_dimensions,
    are_axes_valid,
    lazy_load_generator,
    optimize_threshold,
    reshape_data,
    reshape_data_single,
    reshape_napari,
    get_shape_order,
    REF_AXES,
    NAPARI_AXES
)


###################################################################
# test load_pairs_generator
def test_load_pairs_generator_no_files(tmp_path):
    folders = ['train_X', 'train_Y']

    with pytest.raises(FileNotFoundError):
        load_pairs_generator(tmp_path / folders[0], 'ZXY', tmp_path / folders[1])


def test_load_pairs_generator_unequal_sizes(tmp_path):
    """
    Test that we can load pairs of images with `check_exists` set to `False` when no target exist.

    :param tmp_path:
    :return:
    """
    folders = ['train_X', 'train_Y']
    sizes = [15, 5]
    shapes = [(3, 16, 16) for _ in sizes]

    create_data(tmp_path, folders, sizes, shapes)

    data = load_pairs_generator(tmp_path / folders[0], tmp_path / folders[1], 'ZYX', check_exists=False)

    n = 0
    n_empty = 0
    for source_x, target_y in data.generator():
        n += 1

        if target_y.min() == target_y.max() == 0:
            n_empty += 1
    remove_C_dim

    assert n == sizes[0]
    assert n_empty == sizes[0] - sizes[1]


def test_load_pairs_generator_unequal_sizes_exception(tmp_path):
    """
    Test that with the default `check_exists` parameter set to `True`, loading unmatched pairs generate an exception.

    :param tmp_path:
    :return:
    """
    folders = ['train_X', 'train_Y']
    sizes = [6, 5]
    shapes = [(3, 16, 16) for _ in sizes]

    create_data(tmp_path, folders, sizes, shapes)

    with pytest.raises(FileNotFoundError):
        load_pairs_generator(tmp_path / folders[0], 'ZXY', tmp_path / folders[1])


def test_load_pairs_generator_equal_sizes(tmp_path):
    folders = ['val_X', 'val_Y']
    sizes = [5, 5]
    shapes = [(3, 16, 16) for _ in sizes]

    create_data(tmp_path, folders, sizes, shapes)

    data = load_pairs_generator(tmp_path / folders[0], tmp_path / folders[1], 'ZXY')

    n = 0
    n_empty = 0
    for source_x, target_y in data.generator():
        n += 1

        if target_y.min() == target_y.max() == 0:
            n_empty += 1

    assert n == sizes[0]
    assert n_empty == sizes[0] - sizes[1]


@pytest.mark.parametrize('shape',
                         [(8,), (16, 8), (8, 16, 16), (32, 8, 16, 3), (32, 8, 64, 16, 3), (32, 16, 8, 64, 16, 3)])
def test_load_pairs_generator_dimensions(tmp_path, shape):
    folders = ['val_X', 'val_Y']
    sizes = [5, 5]
    shapes = [(3, 16, 16) for _ in sizes]

    create_data(tmp_path, folders, sizes, shapes)

    load_pairs_generator(tmp_path / folders[0], tmp_path / folders[1], 'ZXY')


###################################################################
# test generate_config
@pytest.mark.parametrize('shape, patch', [((8, 16, 16), (8, 8, 8)), ((1, 1, 8, 16, 16, 1), (8, 8, 8))])
def test_generate_config_wrong_dims(shape, patch):
    """
    Test assertion error from generating a configuration with invalid dimensions (4 > len(dims) or len(dims) > 5).

    :param shape:
    :return:
    """
    with pytest.raises(AssertionError):
        generate_config(np.zeros(shape), patch)


@pytest.mark.parametrize('shape, patch_shape', [((1, 16, 16, 1), (16, 16)),
                                                ((1, 16, 8, 1), (16, 16)),
                                                ((8, 16, 16, 1), (16, 8)),
                                                ((8, 16, 16, 4), (8, 16)),
                                                ((1, 8, 16, 16), (16, 16)),
                                                ((1, 8, 16, 8), (16, 8)),
                                                ((1, 16, 16, 8), (16, 16)),
                                                ((4, 8, 16, 16), (16, 16)),
                                                ((1, 8, 16, 16, 1), (32, 16, 16)),
                                                ((8, 8, 16, 16, 4), (16, 32, 16)),
                                                ((8, 8, 16, 32, 4), (16, 16, 32)),
                                                ((2, 16, 64, 32, 4), (16, 8, 32))])
def test_generate_valid_config(shape, patch_shape):
    """
    Test valid configuration creation for correct shapes (4 <= len(dims) <= 5) and patches.
    :param shape:
    :param patch_shape:
    :return:
    """
    config = generate_config(np.zeros(shape), patch_shape)
    assert config.is_valid()


# TODO test with lists and generator that generate_config works


@pytest.mark.parametrize('shape, patch_shape', [((1, 16, 16, 1), (8, 16, 16)),
                                                ((1, 8, 16, 16, 1), (16, 16))])
def test_generate_config_wrong_patches_shape(shape, patch_shape):
    config = generate_config(np.zeros(shape), patch_shape)
    with pytest.raises(AssertionError):
        assert config.is_valid()


###################################################################
# test build_modelzoo
@pytest.mark.parametrize('shape', [(1, 16, 16, 1),
                                   (1, 16, 8, 1),
                                   (1, 16, 8, 3),
                                   (1, 16, 16, 8, 1),
                                   (1, 16, 16, 8, 3),
                                   (1, 8, 16, 32, 1)])
def test_build_modelzoo_allowed_shapes(tmp_path, shape):
    # create model and save it to disk
    parameters = create_model_zoo_parameters(tmp_path, shape)
    build_modelzoo(*parameters)

    # check if modelzoo exists
    assert Path(parameters[0]).exists()


@pytest.mark.parametrize('shape', [(8,), (8, 16), (1, 16, 16), (32, 16, 8, 16, 32, 3)])
def test_build_modelzoo_disallowed_shapes(tmp_path, shape):
    """
    Test ModelZoo creation based on disallowed shapes.

    :param tmp_path:
    :param shape:
    :return:
    """
    # create model and save it to disk
    with pytest.raises(AssertionError):
        parameters = create_model_zoo_parameters(tmp_path, shape)
        build_modelzoo(*parameters)


@pytest.mark.parametrize('shape', [(8, 16, 16, 3),
                                   (8, 16, 16, 8, 3)])
def test_build_modelzoo_disallowed_batch(tmp_path, shape):
    """
    Test ModelZoo creation based on disallowed shapes.

    :param tmp_path:
    :param shape:
    :return:
    """
    # create model and save it to disk
    with pytest.raises(ValidationError):
        parameters = create_model_zoo_parameters(tmp_path, shape)
        build_modelzoo(*parameters)


###################################################################
# test load_weights
def test_load_weights_wrong_path(tmp_path):
    model = create_model(tmp_path, (1, 16, 16, 1))

    # create a new model and load from previous weights
    model2 = create_model(tmp_path, (1, 16, 16, 1))

    with pytest.raises(FileNotFoundError):
        load_weights(model2, 'definitely not a path')


@pytest.mark.parametrize('shape', [(1, 16, 16, 1), (64, 16, 32, 3), (1, 8, 16, 16, 1), (8, 32, 16, 64, 3)])
def test_load_weights_h5(tmp_path, shape):
    model = create_model(tmp_path, shape)
    path_to_h5 = save_weights_h5(model, tmp_path)

    # create a new model and load previous weights
    model2 = create_model(tmp_path, shape)
    load_weights(model2, str(path_to_h5))


@pytest.mark.parametrize('shape1, shape2', [((1, 16, 16, 1), (1, 8, 16, 16, 1)),
                                            ((1, 8, 16, 16, 1), (1, 16, 16, 1))])
def test_load_weights_h5_incompatible_shapes(tmp_path, shape1, shape2):
    model = create_model(tmp_path, shape1)
    path_to_h5 = save_weights_h5(model, tmp_path)

    # create a new model with different shape
    model2 = create_model(tmp_path, shape2)

    # set previous weights
    with pytest.raises(ValueError):
        load_weights(model2, str(path_to_h5))


@pytest.mark.parametrize('shape', [(1, 16, 16, 1), (8, 16, 32, 3), (1, 8, 16, 16, 1), (1, 8, 16, 16, 1)])
def test_load_weights_modelzoo(tmp_path, shape):
    # save model_zoo
    parameters = create_model_zoo_parameters(tmp_path, shape)
    build_modelzoo(*parameters)

    # create a new model and load from previous weights
    model = create_model(tmp_path, shape)
    load_weights(model, str(parameters[0]))


@pytest.mark.parametrize('shape', [(1, 16, 16, 1), (1, 16, 32, 3), (1, 8, 16, 16, 1), (1, 8, 16, 16, 1)])
def test_load_weights_modelzoo(tmp_path, shape):
    # save model_zoo
    parameters = create_model_zoo_parameters(tmp_path, shape)
    build_modelzoo(*parameters)

    # create a new model and load from previous weights
    model = create_model(tmp_path, shape)
    load_weights(model, str(parameters[0]))


@pytest.mark.parametrize('shape1, shape2', [((1, 16, 16, 1), (1, 8, 16, 16, 1)),
                                            ((1, 8, 16, 16, 1), (1, 16, 16, 1))])
def test_load_weights_h5_incompatible_shapes(tmp_path, shape1, shape2):
    parameters = create_model_zoo_parameters(tmp_path, shape1)
    build_modelzoo(*parameters)

    # create a new model and load from previous weights
    model = create_model(tmp_path, shape2)

    # set previous weights
    with pytest.raises(ValueError):
        load_weights(model, str(parameters[0]))


###################################################################
# test load_from_disk
@pytest.mark.parametrize('shape, axes', [((8, 8), 'YX'), ((4, 8, 8), 'ZYX'), ((5, 8, 8), 'SYX')])
def test_load_from_disk_same_shapes(tmp_path, shape, axes):
    n = 10
    save_img(tmp_path, n, shape)

    # load images
    images = load_from_disk(tmp_path, axes)
    assert type(images) == np.ndarray

    if 'S' in axes:
        assert len(images.shape) == len(shape)
        assert images.shape[0] == n * shape[0]
        assert images.shape[1:] == shape[1:]
    else:
        assert len(images.shape) == len(shape) + 1
        assert images.shape[0] == n
        assert images.shape[1:] == shape

    assert (images[0, ...] != images[1, ...]).all()


@pytest.mark.parametrize('shape1, shape2, axes', [((8, 8), (4, 4), 'YX'),
                                                  ((4, 8, 8), (2, 16, 16), 'ZYX'),
                                                  ((4, 8, 8), (2, 16, 16), 'SYX'),
                                                  ((8, 16), (4, 8, 8), 'YX')])
def test_load_from_disk_different_shapes(tmp_path, shape1, shape2, axes):
    n = [10, 5]
    save_img(tmp_path, n[0], shape1, prefix='im1-')
    save_img(tmp_path, n[1], shape2, prefix='im2-')

    # load images
    images = load_from_disk(tmp_path, axes)
    assert type(images) == tuple
    assert len(images[0]) == n[0] + n[1]

    for img, _ in zip(*images):
        assert (img.shape == shape1) or (img.shape == shape2)


###################################################################
# test load_pairs_from_disk
@pytest.mark.parametrize('shape, axes', [((8, 8), 'YX'),
                                         ((4, 8, 8), 'ZYX'),
                                         ((16, 8, 32, 3), 'ZYXC'),
                                         ((16, 8, 16, 32, 3), 'TZYXC')])
def test_load_pairs_from_disk_same_shape(tmp_path, shape, axes):
    n = 10
    folders = ['X', 'Y']
    for f in folders:
        os.mkdir(tmp_path / f)

    save_img(tmp_path / folders[0], n, shape)

    if 'C' in axes:
        save_img(tmp_path / folders[1], n, shape[:-1])
    else:
        save_img(tmp_path / folders[1], n, shape)

    # load images
    load_pairs_from_disk(tmp_path / folders[0], tmp_path / folders[1], axes)


@pytest.mark.parametrize('shape, axes', [((8, 8), 'YX'),
                                         ((4, 8, 8), 'ZYX'),
                                         ((16, 8, 32, 3), 'ZYXC'),
                                         ((16, 8, 16, 32, 3), 'TZYXC')])
def test_load_pairs_from_disk_different_numbers(tmp_path, shape, axes):
    n = [15, 5]
    folders = ['X', 'Y']
    for f in folders:
        os.mkdir(tmp_path / f)

    save_img(tmp_path / folders[0], n[0], shape)

    if 'C' in axes:
        save_img(tmp_path / folders[1], n[1], shape[:-1])
    else:
        save_img(tmp_path / folders[1], n[1], shape)

    # load images, replacing n[1]-n[0] last images with blank frames
    X, Y, n_loaded = load_pairs_from_disk(tmp_path / folders[0], tmp_path / folders[1], axes, check_exists=False)
    assert Y[n[1]:, ...].min() == Y[n[1]:, ...].max() == 0
    assert n[0] == n_loaded

    # load images with the check_exists flag triggers error
    with pytest.raises(FileNotFoundError):
        load_pairs_from_disk(tmp_path / folders[0], tmp_path / folders[1], axes, check_exists=True)


@pytest.mark.parametrize('shape1, shape2', [((8, 8), (4, 8, 8)), ((4, 8, 8), (8, 8)), ((8, 8), (8, 9))])
def test_load_pairs_from_disk_different_shapes(tmp_path, shape1, shape2):
    n = 10
    folders = ['X', 'Y']
    for f in folders:
        os.mkdir(tmp_path / f)

    save_img(tmp_path / folders[0], n, shape1)
    save_img(tmp_path / folders[1], n, shape2)

    # load images
    with pytest.raises(ValueError):
        load_pairs_from_disk(tmp_path / folders[0], tmp_path / folders[1], 'YX')


@pytest.mark.parametrize('shape1, shape2, axes', [((8, 8, 3), (8, 8), 'YXC'),
                                                  ((4, 8, 8), (8, 8), 'CYX'),
                                                  ((8, 6, 8), (8, 8), 'YCX')])
def test_load_pairs_from_disk_different_shapes_C(tmp_path, shape1, shape2, axes):
    n = 10
    folders = ['X', 'Y']
    for f in folders:
        os.mkdir(tmp_path / f)

    save_img(tmp_path / folders[0], n, shape1)
    save_img(tmp_path / folders[1], n, shape2)

    # load images
    load_pairs_from_disk(tmp_path / folders[0], tmp_path / folders[1], axes)


@pytest.mark.parametrize('shape, axes, final_shape',
                         [((8, 8, 12), 'XYC', (8, 8)),
                          ((4, 8, 8), 'CYX', (8, 8)),
                          ((8, 9, 10), 'YCX', (8, 10)),
                          ((5, 8, 9, 10), 'ZYCX', (5, 8, 10)),
                          ((8, 9, 5, 10), 'CYZX', (9, 5, 10)),
                          ((9, 5, 10), 'YZX', (9, 5, 10))])
def test_remove_C_dim(shape, axes, final_shape):
    new_shape = remove_C_dim(shape, axes)

    if 'C' in axes:
        assert len(new_shape) == len(shape) - 1
    else:
        assert len(new_shape) == len(shape)

    assert new_shape == final_shape


@pytest.mark.parametrize('shape', [3, 4, 5])
@pytest.mark.parametrize('is_3D', [True, False])
def test_filter_dimensions(shape, is_3D):
    permutations = filter_dimensions(shape, is_3D)

    if is_3D:
        assert all(['Z' in p for p in permutations])

    assert all(['YX' == p[-2:] for p in permutations])


def test_filter_dimensions_len6_Z():
    permutations = filter_dimensions(6, True)

    assert all(['Z' in p for p in permutations])
    assert all(['YX' == p[-2:] for p in permutations])


@pytest.mark.parametrize('shape, is_3D', [(2, True), (6, False), (7, True)])
def test_filter_dimensions_error(shape, is_3D):
    permutations = filter_dimensions(shape, is_3D)
    print(permutations)
    assert len(permutations) == 0


@pytest.mark.parametrize('axes, valid', [('XSYCZ', True),
                                         ('YZX', True),
                                         ('TCS', True),
                                         ('xsYcZ', True),
                                         ('YzX', True),
                                         ('tCS', True),
                                         ('SCZXYT', True),
                                         ('SZXCZY', False),
                                         ('Xx', False),
                                         ('SZXGY', False),
                                         ('I5SYX', False),
                                         ('STZCYXL', False)])
def test_are_axes_valid(axes, valid):
    assert are_axes_valid(axes) == valid


def test_lazy_generator(tmp_path):
    n = 10
    save_img(tmp_path, n, (8, 8, 8))

    # create lazy generator
    gen, m = lazy_load_generator(tmp_path)
    assert m == n

    # check that it can load n images
    for i in range(n):
        ret = next(gen, None)
        assert len(ret) == 3
        assert all([r is not None for r in ret])

    # test that next(gen, None) works
    assert next(gen, None) is None

    # test that next() throws error
    with pytest.raises(StopIteration):
        next(gen)


# TODO add more tests
@pytest.mark.parametrize('shape, axes', [((1, 16, 16, 1), 'SYXC')])
def test_optimize_threshold(tmp_path, shape, axes):
    from denoiseg.utils.seg_utils import convert_to_oneHot

    # create model and data
    model = create_model(tmp_path, shape)
    X_val = np.random.random(shape)

    if 'C' in axes:
        y_shape = shape[:-1]
    else:
        y_shape = shape

    Y_val = np.random.randint(0, 255, y_shape, dtype=np.int16)
    _y_onehot = convert_to_oneHot(Y_val)

    # instantiate generator
    gen = optimize_threshold(model, X_val, _y_onehot, axes)

    thresholds = []
    while True:
        t = next(gen, None)

        if t:
            _, temp_threshold, temp_score = t

            thresholds.append(temp_threshold)
        else:
            break

    assert len(thresholds) == 19


############################################
# reshape data
@pytest.mark.parametrize('shape_in, axes_in, final_shape, final_axes',
                         [((16, 8), 'XY', (8, 16), 'YX'),
                          ((16, 5, 8), 'XZY', (5, 8, 16), 'ZYX'),
                          ((16, 5, 8, 12), 'XZYS', (12, 5, 8, 16), 'SZYX'),
                          ((3, 16, 5, 8, 12), 'CXZYS', (12, 5, 8, 16, 3), 'SZYXC'),
                          ((3, 16, 5, 2, 8, 12), 'CXZTYS', (2, 12, 5, 8, 16, 3), 'TSZYXC'),
                          ((2, 16, 5, 8, 12, 3), 'TSZYXC', (2, 16, 5, 8, 12, 3), 'TSZYXC')])
def test_get_shape_order_REF(shape_in, axes_in, final_shape, final_axes):
    new_shape, new_axes, indices = get_shape_order(shape_in, axes_in, REF_AXES)

    assert new_shape == final_shape
    assert new_axes == final_axes

    for i, ind in enumerate(indices):
        assert axes_in[ind] == final_axes[i]


@pytest.mark.parametrize('shape_in, axes_in, final_shape, final_axes',
                         [((16, 8), 'XY', (8, 16), 'YX'),
                          ((16, 5, 8), 'XZY', (5, 8, 16), 'ZYX'),
                          ((16, 5, 8, 12), 'XZYS', (12, 5, 8, 16), 'SZYX'),
                          ((3, 16, 5, 8, 12), 'CXZYS', (3, 12, 5, 8, 16), 'CSZYX'),
                          ((3, 16, 5, 2, 8, 12), 'CXZTYS', (3, 2, 12, 5, 8, 16), 'CTSZYX'),
                          ((2, 16, 5, 8, 12, 3), 'CTSZYX', (2, 16, 5, 8, 12, 3), 'CTSZYX')])
def test_get_shape_order_NAPARI(shape_in, axes_in, final_shape, final_axes):
    new_shape, new_axes, indices = get_shape_order(shape_in, axes_in, NAPARI_AXES)

    assert new_shape == final_shape
    assert new_axes == final_axes

    for i, ind in enumerate(indices):
        assert axes_in[ind] == final_axes[i]


@pytest.mark.parametrize('shape, axes, final_shape, final_axes',
                         [((16, 8), 'YX', (1, 16, 8, 1), 'SYXC'),
                          ((16, 8), 'XY', (1, 8, 16, 1), 'SYXC'),
                          ((16, 3, 8), 'XZY', (1, 3, 8, 16, 1), 'SZYXC'),
                          ((16, 3, 8), 'XYZ', (1, 8, 3, 16, 1), 'SZYXC'),
                          ((16, 3, 8), 'ZXY', (1, 16, 8, 3, 1), 'SZYXC'),
                          ((16, 3, 12), 'SXY', (16, 12, 3, 1), 'SYXC'),
                          ((5, 5, 2), 'XYS', (2, 5, 5, 1), 'SYXC'),
                          ((5, 1, 5, 2), 'XZYS', (2, 1, 5, 5, 1), 'SZYXC'),
                          ((5, 12, 5, 2), 'ZXYS', (2, 5, 5, 12, 1), 'SZYXC'),
                          ((16, 8, 5, 12), 'SZYX', (16, 8, 5, 12, 1), 'SZYXC')])
def test_reshape_data_no_CT(shape, axes, final_shape, final_axes):
    x = np.zeros(shape)
    y = np.zeros(shape)
    final_shape_y = final_shape[:-1]

    _x, _y, new_axes = reshape_data(x, y, axes)

    assert _x.shape == final_shape
    assert _y.shape == final_shape_y
    assert new_axes == final_axes


@pytest.mark.parametrize('shape, axes, final_shape, final_axes',
                         [((16, 8, 5), 'YXT', (5, 16, 8, 1), 'SYXC'),
                          ((4, 16, 8), 'TXY', (4, 8, 16, 1), 'SYXC'),
                          ((4, 16, 6, 8), 'TXSY', (4 * 6, 8, 16, 1), 'SYXC'),
                          ((4, 16, 6, 5, 8), 'ZXTYS', (8 * 6, 4, 5, 16, 1), 'SZYXC')])
def test_reshape_data_T_no_C(shape, axes, final_shape, final_axes):
    x = np.zeros(shape)
    y = np.zeros(shape)
    final_shape_y = final_shape[:-1]

    _x, _y, new_axes = reshape_data(x, y, axes)

    assert _x.shape == final_shape
    assert _y.shape == final_shape_y
    assert new_axes == final_axes


@pytest.mark.parametrize('shape, axes, final_shape, final_axes',
                         [((5, 3, 5), 'XCY', (1, 5, 5, 3), 'SYXC'),
                          ((16, 3, 12, 8), 'XCYS', (8, 12, 16, 3), 'SYXC'),
                          ((16, 3, 12, 8), 'ZXCY', (1, 16, 8, 3, 12), 'SZYXC'),
                          ((16, 3, 12, 8), 'XCYZ', (1, 8, 12, 16, 3), 'SZYXC'),
                          ((16, 3, 12, 8), 'ZYXC', (1, 16, 3, 12, 8), 'SZYXC'),
                          ((16, 3, 21, 12, 8), 'ZYSXC', (21, 16, 3, 12, 8), 'SZYXC'),
                          ((16, 3, 21, 8, 12), 'SZYCX', (16, 3, 21, 12, 8), 'SZYXC')])
def test_reshape_data_C_no_T(shape, axes, final_shape, final_axes):
    x = np.zeros(shape)

    # Y does not have C dimension
    ind_c = axes.find('C')
    shape_y = list(shape)
    shape_y[ind_c] = 1
    y = np.zeros(shape_y).squeeze()  # Remove C dimension
    final_shape_y = final_shape[:-1]

    assert len(x.shape) == len(y.shape) + 1

    _x, _y, new_axes = reshape_data(x, y, axes)

    assert _x.shape == final_shape
    assert _y.shape == final_shape_y
    assert new_axes == final_axes


@pytest.mark.parametrize('shape, axes, final_shape, final_axes',
                         [((5, 3, 8, 6), 'XTCY', (3, 6, 5, 8), 'SYXC'),
                          ((16, 3, 12, 5, 8), 'XCYTS', (8 * 5, 12, 16, 3), 'SYXC'),
                          ((16, 10, 5, 6, 12, 8), 'ZSXCYT', (10 * 8, 16, 12, 5, 6), 'SZYXC')])
def test_reshape_data_CT(shape, axes, final_shape, final_axes):
    x = np.zeros(shape)

    # Y does not have C dimension
    ind_c = axes.find('C')
    shape_y = list(shape)
    shape_y[ind_c] = 1
    y = np.zeros(shape_y).squeeze()  # Remove C dimension
    final_shape_y = final_shape[:-1]

    assert len(x.shape) == len(y.shape) + 1

    _x, _y, new_axes = reshape_data(x, y, axes)

    assert _x.shape == final_shape
    assert _y.shape == final_shape_y
    assert new_axes == final_axes


def test_reshape_data_values_XY():
    # test if X and Y are flipped
    axes = 'XY'
    shape = (16, 8)
    x = np.random.randint(0, 255, shape)
    y = np.random.randint(0, 255, shape)

    _x, _y, _ = reshape_data(x, y, axes)
    _x = _x.squeeze()
    _y = _y.squeeze()

    for i in range(shape[1]):
        assert (_x[i, :] == x[:, i]).all()
        assert (_y[i, :] == y[:, i]).all()


def test_reshape_data_values_CXY():
    # test if X and Y are flipped
    axes = 'CXY'
    shape = (3, 16, 8)
    x = np.random.randint(0, 255, shape)
    y = np.random.randint(0, 255, shape[1:])

    _x, _y, _ = reshape_data(x, y, axes)
    _x = _x.squeeze()
    _y = _y.squeeze()

    for i in range(shape[2]):
        for c in range(shape[0]):
            assert (_x[i, :, c] == x[c, :, i]).all()
        assert (_y[i, :] == y[:, i]).all()


def test_reshape_data_values_XZY():
    axes = 'XZY'
    shape = (16, 5, 8)
    x = np.random.randint(0, 255, shape)
    y = np.random.randint(0, 255, shape)

    _x, _y, _ = reshape_data(x, y, axes)
    _x = _x.squeeze()
    _y = _y.squeeze()

    for z in range(shape[1]):
        for i in range(shape[2]):
            assert (_x[z, i, :] == x[:, z, i]).all()
            assert (_y[z, i, :] == y[:, z, i]).all()


def test_reshape_data_values_XZTY():
    axes = 'XZTY'
    shape = (16, 15, 5, 8)
    x = np.random.randint(0, 255, shape)
    y = np.random.randint(0, 255, shape)

    _x, _y, _ = reshape_data(x, y, axes)
    _x = _x.squeeze()
    _y = _y.squeeze()

    for t in range(shape[2]):
        for z in range(shape[1]):
            for i in range(shape[3]):
                assert (_x[t, z, i, :] == x[:, z, t, i]).all()
                assert (_y[t, z, i, :] == y[:, z, t, i]).all()


def test_reshape_data_values_STYX():
    axes = 'STYX'
    shape = (5, 10, 16, 8)
    x = np.random.randint(0, 255, shape)
    y = np.random.randint(0, 255, shape)

    _x, _y, _ = reshape_data(x, y, axes)
    _x = _x.squeeze()

    for s in range(shape[0]):
        for t in range(shape[1]):
            for i in range(shape[2]):
                # here reshaping happens because S and T dims are pulled together
                assert (_x[t * shape[0] + s, i, :] == x[s, t, i, :]).all()
                assert (_y[t * shape[0] + s, i, :] == y[s, t, i, :]).all()


def test_reshape_data_values_TSYX():
    axes = 'TSYX'
    shape = (5, 10, 16, 8)
    x = np.random.randint(0, 255, shape)
    y = np.random.randint(0, 255, shape)

    _x, _y, _ = reshape_data(x, y, axes)
    _x = _x.squeeze()

    for s in range(shape[1]):
        for t in range(shape[0]):
            for i in range(shape[2]):
                # here reshaping happens because S and T dims are pulled together
                assert (_x[t * shape[1] + s, i, :] == x[t, s, i, :]).all()
                assert (_y[t * shape[1] + s, i, :] == y[t, s, i, :]).all()


def test_reshape_data_values_SZYTX():
    axes = 'ZSYTX'
    shape = (15, 10, 16, 5, 8)
    x = np.random.randint(0, 255, shape)
    y = np.random.randint(0, 255, shape)

    _x, _y, _ = reshape_data(x, y, axes)
    _x = _x.squeeze()

    for s in range(shape[1]):
        for t in range(shape[3]):
            for z in range(shape[0]):
                for i in range(shape[2]):
                    # here reshaping happens because S and T dims are pulled together
                    assert (_x[t * shape[1] + s, z, i, :] == x[z, s, i, t, :]).all()
                    assert (_y[t * shape[1] + s, z, i, :] == y[z, s, i, t, :]).all()


def test_reshape_data_values_ZTYSX():
    axes = 'ZTYSX'
    shape = (15, 10, 16, 5, 8)
    x = np.random.randint(0, 255, shape)
    y = np.random.randint(0, 255, shape)

    _x, _y, _ = reshape_data(x, y, axes)
    _x = _x.squeeze()

    for s in range(shape[3]):
        for t in range(shape[1]):
            for z in range(shape[0]):
                for i in range(shape[2]):
                    # here reshaping happens because S and T dims are pulled together
                    assert (_x[t * shape[3] + s, z, i, :] == x[z, t, i, s, :]).all()
                    assert (_y[t * shape[3] + s, z, i, :] == y[z, t, i, s, :]).all()


def test_reshape_data_values_ZTCYSX():
    axes = 'ZTCYSX'
    shape = (15, 10, 3, 16, 5, 8)
    x = np.random.randint(0, 255, shape)
    y = np.random.randint(0, 255, (*shape[:2], *shape[3:]))  # no C dimension

    _x, _y, _ = reshape_data(x, y, axes)

    for s in range(shape[4]):
        for t in range(shape[1]):
            for z in range(shape[0]):
                for i in range(shape[3]):
                    for c in range(shape[2]):
                        # here reshaping happens because S and T dims are pulled together
                        assert (_x[t * shape[4] + s, z, i, :, c] == x[z, t, c, i, s, :]).all()

                    assert (_y[t * shape[4] + s, z, i, :] == y[z, t, i, s, :]).all()


##########################################
# reshape data single
@pytest.mark.parametrize('shape, axes, final_shape, final_axes',
                         [((16, 8), 'YX', (1, 16, 8, 1), 'SYXC'),
                          ((16, 8), 'XY', (1, 8, 16, 1), 'SYXC'),
                          ((16, 3, 8), 'XZY', (1, 3, 8, 16, 1), 'SZYXC'),
                          ((16, 3, 8), 'XYZ', (1, 8, 3, 16, 1), 'SZYXC'),
                          ((16, 3, 8), 'ZXY', (1, 16, 8, 3, 1), 'SZYXC'),
                          ((16, 3, 12), 'SXY', (16, 12, 3, 1), 'SYXC'),
                          ((5, 5, 2), 'XYS', (2, 5, 5, 1), 'SYXC'),
                          ((5, 1, 5, 2), 'XZYS', (2, 1, 5, 5, 1), 'SZYXC'),
                          ((5, 12, 5, 2), 'ZXYS', (2, 5, 5, 12, 1), 'SZYXC'),
                          ((16, 8, 5, 12), 'SZYX', (16, 8, 5, 12, 1), 'SZYXC'),
                          ((16, 8, 5), 'YXT', (5, 16, 8, 1), 'SYXC'),  # T, no C
                          ((4, 16, 8), 'TXY', (4, 8, 16, 1), 'SYXC'),
                          ((4, 16, 6, 8), 'TXSY', (4 * 6, 8, 16, 1), 'SYXC'),
                          ((4, 16, 6, 5, 8), 'ZXTYS', (8 * 6, 4, 5, 16, 1), 'SZYXC'),
                          ((5, 3, 5), 'XCY', (1, 5, 5, 3), 'SYXC'),  # C, no T
                          ((16, 3, 12, 8), 'XCYS', (8, 12, 16, 3), 'SYXC'),
                          ((16, 3, 12, 8), 'ZXCY', (1, 16, 8, 3, 12), 'SZYXC'),
                          ((16, 3, 12, 8), 'XCYZ', (1, 8, 12, 16, 3), 'SZYXC'),
                          ((16, 3, 12, 8), 'ZYXC', (1, 16, 3, 12, 8), 'SZYXC'),
                          ((16, 3, 21, 12, 8), 'ZYSXC', (21, 16, 3, 12, 8), 'SZYXC'),
                          ((16, 3, 21, 8, 12), 'SZYCX', (16, 3, 21, 12, 8), 'SZYXC'),
                          ((5, 3, 8, 6), 'XTCY', (3, 6, 5, 8), 'SYXC'),  # CT
                          ((16, 3, 12, 5, 8), 'XCYTS', (8 * 5, 12, 16, 3), 'SYXC'),
                          ((16, 10, 5, 6, 12, 8), 'ZSXCYT', (10 * 8, 16, 12, 5, 6), 'SZYXC')
                          ])
def test_reshape_data_single(shape, axes, final_shape, final_axes):
    x = np.zeros(shape)

    _x, new_axes = reshape_data_single(x, axes)

    assert _x.shape == final_shape
    assert new_axes == final_axes


def test_reshape_single_data_values_XY():
    # test if X and Y are flipped
    axes = 'XY'
    shape = (16, 8)
    x = np.random.randint(0, 255, shape)

    _x, _ = reshape_data_single(x, axes)
    _x = _x.squeeze()

    for i in range(shape[1]):
        assert (_x[i, :] == x[:, i]).all()


def test_reshape_single_data_values_XZY():
    axes = 'XZY'
    shape = (16, 5, 8)
    x = np.random.randint(0, 255, shape)

    _x, _ = reshape_data_single(x, axes)
    _x = _x.squeeze()

    for z in range(shape[1]):
        for i in range(shape[2]):
            assert (_x[z, i, :] == x[:, z, i]).all()


def test_reshape_single_data_values_XZTY():
    axes = 'XZTY'
    shape = (16, 15, 5, 8)
    x = np.random.randint(0, 255, shape)

    _x, _ = reshape_data_single(x, axes)
    _x = _x.squeeze()

    for t in range(shape[2]):
        for z in range(shape[1]):
            for i in range(shape[3]):
                assert (_x[t, z, i, :] == x[:, z, t, i]).all()


def test_reshape_single_data_values_STYX():
    axes = 'STYX'
    shape = (5, 10, 16, 8)
    x = np.random.randint(0, 255, shape)

    _x, _ = reshape_data_single(x, axes)
    _x = _x.squeeze()

    for s in range(shape[0]):
        for t in range(shape[1]):
            for i in range(shape[2]):
                # here reshaping happens because S and T dims are pulled together
                assert (_x[t * shape[0] + s, i, :] == x[s, t, i, :]).all()


def test_reshape_single_data_values_TSYX():
    axes = 'TSYX'
    shape = (5, 10, 16, 8)
    x = np.random.randint(0, 255, shape)

    _x, _ = reshape_data_single(x, axes)
    _x = _x.squeeze()

    for s in range(shape[1]):
        for t in range(shape[0]):
            for i in range(shape[2]):
                # here reshaping happens because S and T dims are pulled together
                assert (_x[t * shape[1] + s, i, :] == x[t, s, i, :]).all()


def test_reshape_single_data_values_SZYTX():
    axes = 'ZSYTX'
    shape = (15, 10, 16, 5, 8)
    x = np.random.randint(0, 255, shape)

    _x, _ = reshape_data_single(x, axes)
    _x = _x.squeeze()

    for s in range(shape[1]):
        for t in range(shape[3]):
            for z in range(shape[0]):
                for i in range(shape[2]):
                    # here reshaping happens because S and T dims are pulled together
                    assert (_x[t * shape[1] + s, z, i, :] == x[z, s, i, t, :]).all()


def test_reshape_single_data_values_ZTYSX():
    axes = 'ZTYSX'
    shape = (15, 10, 16, 5, 8)
    x = np.random.randint(0, 255, shape)

    _x, _ = reshape_data_single(x, axes)
    _x = _x.squeeze()

    for s in range(shape[3]):
        for t in range(shape[1]):
            for z in range(shape[0]):
                for i in range(shape[2]):
                    # here reshaping happens because S and T dims are pulled together
                    assert (_x[t * shape[3] + s, z, i, :] == x[z, t, i, s, :]).all()


def test_reshape_single_data_values_SZCYTX():
    axes = 'ZTCYSX'
    shape = (15, 10, 3, 16, 5, 8)
    x = np.random.randint(0, 255, shape)

    _x, _ = reshape_data_single(x, axes)

    for s in range(shape[4]):
        for t in range(shape[1]):
            for z in range(shape[0]):
                for c in range(shape[2]):
                    for i in range(shape[3]):
                        # here reshaping happens because S and T dims are pulled together
                        assert (_x[t * shape[4] + s, z, i, :, c] == x[z, t, c, i, s, :]).all()


##########################################
# reshape napari
def test_reshape_data_napari_values_XY():
    # test if X and Y are flipped
    axes = 'XY'
    shape = (16, 8)
    x = np.random.randint(0, 255, shape)

    _x, _ = reshape_napari(x, axes)

    for i in range(shape[1]):
        assert (_x[i, :] == x[:, i]).all()


def test_reshape_data_napari_values_XZY():
    # test if X and Y are flipped
    axes = 'XZY'
    shape = (16, 5, 8)
    x = np.random.randint(0, 255, shape)

    _x, _ = reshape_napari(x, axes)

    for z in range(shape[1]):
        for i in range(shape[2]):
            assert (_x[z, i, :] == x[:, z, i]).all()


def test_reshape_data_napari_values_SYXC():
    # test if X and Y are flipped
    axes = 'SYXC'
    shape = (10, 16, 8, 3)
    x = np.random.randint(0, 255, shape)

    _x, _ = reshape_napari(x, axes)

    for c in range(shape[-1]):
        for s in range(shape[0]):
            for i in range(shape[1]):
                assert (_x[c, s, i, :] == x[s, i, :, c]).all()


def test_reshape_data_napari_values_SZYXC():
    # test if X and Y are flipped
    axes = 'SZYXC'
    shape = (10, 15, 16, 8, 3)
    x = np.random.randint(0, 255, shape)

    _x, _ = reshape_napari(x, axes)

    for c in range(shape[-1]):
        for s in range(shape[0]):
            for z in range(shape[1]):
                for i in range(shape[2]):
                    assert (_x[c, s, z, i, :] == x[s, z, i, :, c]).all()


def test_reshape_data_napari_values_SZYXC():
    # test if X and Y are flipped
    axes = 'YSXTZC'
    shape = (16, 10, 8, 15, 3, 2)
    x = np.random.randint(0, 255, shape)

    _x, _ = reshape_napari(x, axes)

    for c in range(shape[5]):
        for t in range(shape[3]):
            for s in range(shape[1]):
                for z in range(shape[4]):
                    for i in range(shape[0]):
                        assert (_x[c, t, s, z, i, :] == x[i, s, :, t, z, c]).all()


@pytest.mark.parametrize('shape, axes, final_shape, final_axes',
                         [((16, 8), 'YX', (16, 8), 'YX'),
                          ((16, 8), 'XY', (8, 16), 'YX'),
                          ((16, 8, 5), 'XYZ', (5, 8, 16), 'ZYX'),
                          ((5, 16, 8), 'ZXY', (5, 8, 16), 'ZYX'),
                          ((12, 16, 8, 10), 'TXYS', (12, 10, 8, 16), 'TSYX'),
                          ((10, 5, 16, 8, 3), 'SZXYC', (3, 10, 5, 8, 16), 'CSZYX'),
                          ((16, 10, 3, 8), 'YSCX', (3, 10, 16, 8), 'CSYX')
                          ])
def test_reshape_data_napari(shape, axes, final_shape, final_axes):
    x = np.zeros(shape)

    _x, new_axes = reshape_napari(x, axes)

    assert _x.shape == final_shape
    assert new_axes == final_axes
