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
    optimize_threshold
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
    assert type(images) == list
    assert len(images) == n[0] + n[1]

    for img in images:
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


@pytest.mark.parametrize('shape, axes', [((1, 16, 16, 1), 'SYXC')])
def test_optimize_threshold(tmp_path, shape, axes):
    # create model and data
    model = create_model(tmp_path, shape)
    X_val = np.random.random(shape)
    Y_val = np.random.randint(0, 255, shape, dtype=np.uint16)

    # instantiate generator
    gen = optimize_threshold(model, X_val, Y_val, axes)

    thresholds = []
    while True:
        t = next(gen, None)

        if t:
            _, temp_threshold, temp_score = t

            thresholds.append(temp_threshold)
        else:
            break

    assert len(thresholds) == 19
