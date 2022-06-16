from pathlib import Path
import numpy as np
import pytest

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
    load_from_disk
)


###################################################################
# test from_folder
def test_from_folder_no_files(tmp_path):
    folders = ['train_X', 'train_Y']

    with pytest.raises(FileNotFoundError):
        from_folder(tmp_path / folders[0], tmp_path / folders[1])


def test_from_folder_unequal_sizes(tmp_path):
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


###################################################################
# test generate_config
@pytest.mark.parametrize('shape', [(8, 16, 16), (1, 1, 8, 16, 16, 1)])
def test_dimensions_generate_config_exception(shape):
    # <4 and >5 dimensions are not accepted: ZXY
    with pytest.raises(AssertionError):
        generate_config(np.zeros(shape))


@pytest.mark.parametrize('shape', [(1, 16, 16, 1),
                                   (8, 16, 16, 1),
                                   (8, 16, 16, 4),
                                   (1, 8, 16, 16),
                                   (4, 8, 16, 16),
                                   (1, 8, 16, 16, 1),
                                   (8, 8, 16, 16, 4)])
def test_dimensions_generate_config_no_exception(shape):
    # 4 and 5 are accepted: SYXC or SZYXC, where S is batch and C are channels (eg. RGB)
    generate_config(np.zeros(shape))


###################################################################
# test build_modelzoo
# TODO this should change once 3D is enabled
@pytest.mark.parametrize('shape', [(1, 16, 16, 1)])
def test_build_modelzoo_allowed_shapes(tmp_path, shape):
    # create model and save it to disk
    parameters = create_model_zoo_parameters(tmp_path, shape)
    build_modelzoo(*parameters)

    # check if modelzoo exists
    assert Path(parameters[0]).exists()


# TODO this should change once 3D is enabled
@pytest.mark.parametrize('shape', [(1, 16, 16), (1, 8, 16, 16, 1)])
def test_build_modelzoo_disallowed_shapes(tmp_path, shape):
    # create model and save it to disk
    with pytest.raises(AssertionError):
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


@pytest.mark.parametrize('shape', [(1, 16, 16, 1), (1, 8, 16, 16, 1)])
def test_load_weights_h5(tmp_path, shape):
    model = create_model(tmp_path, shape)
    path_to_h5 = save_weights_h5(model, tmp_path)

    # create a new model and load from previous weights
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


# TODO: extend tests to 3D when functionality will be here
def test_load_weights_modelzoo(tmp_path):
    shape = (1, 8, 8, 1)

    # save model_zoo
    parameters = create_model_zoo_parameters(tmp_path, shape)
    build_modelzoo(*parameters)

    # create a new model and load from previous weights
    model = create_model(tmp_path, shape)
    load_weights(model, str(parameters[0]))


# TODO: make it a test once 3D functionality is here
@pytest.mark.parametrize('shape1, shape2', [((1, 16, 16, 1), (1, 8, 16, 16, 1)),
                                            ((1, 8, 16, 16, 1), (1, 16, 16, 1))])
def load_weights_h5_incompatible_shapes(tmp_path, shape1, shape2):
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
def test_load_from_disk_equal_sizes(tmp_path, shape):
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
def test_load_from_disk_unequal_sizes(tmp_path, shape1, shape2):
    n = [10, 5]
    save_img(tmp_path, n[0], shape1, prefix='im1-')
    save_img(tmp_path, n[1], shape2, prefix='im2-')

    # load images
    images = load_from_disk(tmp_path)
    assert type(images) == list
    assert len(images) == n[0]+n[1]

    for img in images:
        assert (img.shape == shape1) or (img.shape == shape2)


# TODO: missing load_pairs_from_disk tests
