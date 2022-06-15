import os
from pathlib import Path
import numpy as np
import pytest
from tifffile import imwrite

from denoiseg.models import DenoiSeg
from napari_denoiseg.utils import (
    from_folder,
    generate_config,
    build_modelzoo,
    load_weights,
    load_from_disk,
    load_pairs_from_disk
)


###################################################################
# convenience functions: save images
def save_img(folder_path, n, shape, prefix=''):
    for i in range(n):
        im = np.random.randint(0, 65535, shape, dtype=np.uint16)
        imwrite(os.path.join(folder_path, prefix + str(i) + '.tif'), im)


def create_data(main_dir, folders, sizes, shape):
    for n, f in zip(sizes, folders):
        source = main_dir / f
        os.mkdir(source)
        save_img(source, n, shape)


def test_create_data(tmp_path):
    folders = ['train_X', 'train_Y', 'val_X', 'val_Y']
    sizes = [20, 8, 5, 5]

    create_data(tmp_path, folders, sizes, (1, 8, 16, 16))

    for n, f in zip(sizes, folders):
        source = tmp_path / f
        files = [f for f in Path(source).glob('*.tif*')]
        assert len(files) == n


# convenience functions: create and save models
def create_model(basedir, shape):
    # create model
    X = np.zeros(shape)
    name = 'myModel'
    config = generate_config(X)

    return DenoiSeg(config, name, basedir)


def save_weights_h5(model, basedir):
    name_weights = 'myModel.h5'
    path_to_weights = basedir / name_weights

    # save model
    model.keras_model.save_weights(path_to_weights)

    return path_to_weights


def test_saved_weights_h5(tmp_path):
    model = create_model(tmp_path, (1, 16, 16, 1))
    path_to_weights = save_weights_h5(model, tmp_path)

    assert path_to_weights.name.endswith('.h5')
    assert path_to_weights.exists()


# TODO: why is it saving in the current directory and not in folder?
def create_model_zoo(folder, shape):
    # create model and save it to disk
    model = create_model(folder, shape)
    path_to_h5 = str(save_weights_h5(model, folder).resolve())

    # path to modelzoo
    path_to_modelzoo = path_to_h5[:-len('.h5')] + '.bioimage.io.zip'

    # inputs/outputs
    path_to_input = path_to_h5[:-len('.h5')] + 'input.npy'
    np.save(path_to_input, np.zeros(shape))
    assert Path(path_to_input).exists()

    path_to_output = path_to_h5[:-len('.h5')] + 'output.npy'
    np.save(path_to_output, np.zeros(shape))
    assert Path(path_to_output).exists()

    # documentation
    path_to_doc = folder / 'doc.md'
    with open(path_to_doc, 'w') as f:
        pass
    assert path_to_doc.exists()

    # build modelzoo
    tf_version = 42
    build_modelzoo(path_to_modelzoo, path_to_h5, path_to_input, path_to_output, tf_version, path_to_doc)

    return path_to_modelzoo


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
    path_to_modelzoo = create_model_zoo(tmp_path, shape)

    # check if modelzoo exists
    assert Path(path_to_modelzoo).exists()


# TODO this should change once 3D is enabled
@pytest.mark.parametrize('shape', [(1, 16, 16), (1, 8, 16, 16, 1)])
def test_build_modelzoo_disallowed_shapes(tmp_path, shape):
    # create model and save it to disk
    with pytest.raises(AssertionError):
        path_to_modelzoo = create_model_zoo(tmp_path, shape)


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
    path_to_modelzoo = create_model_zoo(tmp_path, shape)

    # create a new model and load from previous weights
    model = create_model(tmp_path, shape)
    load_weights(model, str(path_to_modelzoo))


# TODO: make it a test once 3D functionality is here
@pytest.mark.parametrize('shape1, shape2', [((1, 16, 16, 1), (1, 8, 16, 16, 1)),
                                            ((1, 8, 16, 16, 1), (1, 16, 16, 1))])
def load_weights_h5_incompatible_shapes(tmp_path, shape1, shape2):
    path_to_modelzoo = create_model_zoo(tmp_path, shape1)

    # create a new model and load from previous weights
    model = create_model(tmp_path, shape2)

    # set previous weights
    with pytest.raises(ValueError):
        load_weights(model, str(path_to_modelzoo))


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
