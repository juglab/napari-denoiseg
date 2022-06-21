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
    from_folder,
    generate_config,
    build_modelzoo,
    load_weights,
    load_from_disk,
    load_pairs_from_disk
)


###################################################################
# test from_folder
def test_from_folder_no_files(tmp_path):
    folders = ['train_X', 'train_Y']

    with pytest.raises(FileNotFoundError):
        from_folder(tmp_path / folders[0], tmp_path / folders[1])


def test_from_folder_unequal_sizes(tmp_path):
    """
    Test that we can load pairs of images with `check_exists` set to `False` when no target exist.

    :param tmp_path:
    :return:
    """
    folders = ['train_X', 'train_Y']
    sizes = [15, 5]

    create_data(tmp_path, folders, sizes, (3, 16, 16))

    data = from_folder(tmp_path / folders[0], tmp_path / folders[1], check_exists=False)

    n = 0
    n_empty = 0
    for source_x, target_y, _, _ in data.generator():
        n += 1

        if target_y.min() == target_y.max() == 0:
            n_empty += 1

    assert n == sizes[0]
    assert n_empty == sizes[0] - sizes[1]


def test_from_folder_unequal_sizes_exception(tmp_path):
    """
    Test that with the default `check_exists` parameter set to `True`, loading unmatched pairs generate an exception.

    :param tmp_path:
    :return:
    """
    folders = ['train_X', 'train_Y']
    sizes = [6, 5]

    create_data(tmp_path, folders, sizes, (3, 16, 16))

    with pytest.raises(FileNotFoundError):
        from_folder(tmp_path / folders[0], tmp_path / folders[1])


def test_from_folder_equal_sizes(tmp_path):
    folders = ['val_X', 'val_Y']
    sizes = [5, 5]

    create_data(tmp_path, folders, sizes, (3, 16, 16))

    data = from_folder(tmp_path / folders[0], tmp_path / folders[1])

    n = 0
    n_empty = 0
    for source_x, target_y, _, _ in data.generator():
        n += 1

        if target_y.min() == target_y.max() == 0:
            n_empty += 1

    assert n == sizes[0]
    assert n_empty == sizes[0] - sizes[1]


@pytest.mark.parametrize('shape', [(8,), (16, 8), (8, 16, 16), (32, 8, 16, 3), (32, 8, 64, 16, 3), (32, 16, 8, 64, 16, 3)])
def test_from_folder_dimensions(tmp_path, shape):
    folders = ['val_X', 'val_Y']
    sizes = [5, 5]

    create_data(tmp_path, folders, sizes, (3, 16, 16))

    from_folder(tmp_path / folders[0], tmp_path / folders[1])


###################################################################
# test generate_config
@pytest.mark.parametrize('shape', [(8, 16, 16), (1, 1, 8, 16, 16, 1)])
def test_generate_config__wrong_dims(shape):
    """
    Test assertion error from generating a configuration with invalid dimensions (4 > len(dims) or len(dims) > 5).

    :param shape:
    :return:
    """
    with pytest.raises(AssertionError):
        generate_config(np.zeros(shape))


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
    path_to_h5 = save_weights_h5(model, tmp_path)

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
@pytest.mark.parametrize('shape', [(8, 8), (4, 8, 8)])
def test_load_from_disk_same_shapes(tmp_path, shape):
    n = 10
    save_img(tmp_path, n, shape)

    # load images
    images = load_from_disk(tmp_path)
    assert type(images) == np.ndarray
    assert len(images.shape) == len(shape)+1
    assert images.shape[0] == n
    assert images.shape[1:] == shape
    assert (images[0, ...] != images[1, ...]).all()


@pytest.mark.parametrize('shape1, shape2', [((8, 8), (4, 4)), ((4, 8, 8), (2, 16, 16)), ((8, 16), (4, 8, 8))])
def test_load_from_disk_different_shapes(tmp_path, shape1, shape2):
    n = [10, 5]
    save_img(tmp_path, n[0], shape1, prefix='im1-')
    save_img(tmp_path, n[1], shape2, prefix='im2-')

    # load images
    images = load_from_disk(tmp_path)
    assert type(images) == list
    assert len(images) == n[0]+n[1]

    for img in images:
        assert (img.shape == shape1) or (img.shape == shape2)


###################################################################
# test load_pairs_from_disk
@pytest.mark.parametrize('shape', [(8, 8), (4, 8, 8), (16, 8, 32, 3), (16, 8, 16, 32, 3)])
def test_load_pairs_from_disk_same_shape(tmp_path, shape):
    n = 10
    folders = ['X', 'Y']
    for f in folders:
        os.mkdir(tmp_path / f)

    save_img(tmp_path / folders[0], n, shape)
    save_img(tmp_path / folders[1], n, shape)

    # load images
    X, Y = load_pairs_from_disk(tmp_path / folders[0], tmp_path / folders[1])


@pytest.mark.parametrize('shape', [(8, 8), (4, 8, 8), (16, 8, 32, 3), (16, 8, 16, 32, 3)])
def test_load_pairs_from_disk_different_numbers(tmp_path, shape):
    n = [15, 5]
    folders = ['X', 'Y']
    for f in folders:
        os.mkdir(tmp_path / f)

    save_img(tmp_path / folders[0], n[0], shape)
    save_img(tmp_path / folders[1], n[1], shape)

    # load images, replacing n[1]-n[0] last images with blank frames
    X, Y = load_pairs_from_disk(tmp_path / folders[0], tmp_path / folders[1], check_exists=False)
    assert Y[n[1]:, ...].min() == Y[n[1]:, ...].max() == 0

    # load images with the check_exists flag triggers error
    with pytest.raises(FileNotFoundError):
        load_pairs_from_disk(tmp_path / folders[0], tmp_path / folders[1], check_exists=True)


@pytest.mark.parametrize('shape1, shape2', [((8, 8), (4, 8, 8)), ((4, 8, 8), (8, 8)), ((8, 8), (8, 9))])
def test_load_pairs_from_disk_different_shapes(tmp_path, shape1, shape2):
    n = 10
    folders = ['X', 'Y']
    for f in folders:
        os.mkdir(tmp_path / f)

    save_img(tmp_path / folders[0], n, shape1)
    save_img(tmp_path / folders[1], n, shape2)

    # load images
    with pytest.raises(AssertionError):
        load_pairs_from_disk(tmp_path / folders[0], tmp_path / folders[1])

