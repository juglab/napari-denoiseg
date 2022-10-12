from pathlib import Path

import pytest
import numpy as np

from napari_denoiseg._tests.test_utils import (
    save_weights_h5,
    create_model,
    create_model_zoo_parameters
)
from napari_denoiseg.utils import (
    build_modelzoo,
    generate_config,
    load_configuration,
    save_configuration,
    load_weights,
    cwd
)


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


@pytest.mark.parametrize('shape, patch_shape', [((1, 16, 16, 1), (16, 16)),
                                                ((1, 16, 16, 16, 1), (16, 16, 16))])
def test_save_configuration(tmp_path, shape, patch_shape):
    config = generate_config(np.zeros(shape), patch_shape)

    # sanity check
    assert config.is_valid()

    # save config
    save_configuration(config, tmp_path)

    # check if exists
    assert Path(tmp_path / 'config.json').exists()


@pytest.mark.parametrize('shape, patch_shape', [((1, 16, 16, 1), (16, 16)),
                                                ((1, 16, 16, 16, 1), (16, 16, 16))])
def test_load_configuration(tmp_path, shape, patch_shape):
    config = generate_config(np.zeros(shape), patch_shape)

    # sanity check
    assert config.is_valid()

    # save config
    save_configuration(config, tmp_path)

    # load config
    config_loaded = load_configuration(tmp_path / 'config.json')
    assert config_loaded.is_valid()


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

    with cwd(tmp_path):
        build_modelzoo(*parameters)

    # create a new model and load from previous weights
    model = create_model(tmp_path, shape)
    load_weights(model, str(parameters[0]))


@pytest.mark.parametrize('shape', [(1, 16, 16, 1), (1, 16, 32, 3), (1, 8, 16, 16, 1), (1, 8, 16, 16, 1)])
def test_load_weights_modelzoo(tmp_path, shape):
    # save model_zoo
    parameters = create_model_zoo_parameters(tmp_path, shape)

    with cwd(tmp_path):
        build_modelzoo(*parameters)

    # create a new model and load from previous weights
    model = create_model(tmp_path, shape)
    load_weights(model, str(parameters[0]))


@pytest.mark.parametrize('shape1, shape2', [((1, 16, 16, 1), (1, 8, 16, 16, 1)),
                                            ((1, 8, 16, 16, 1), (1, 16, 16, 1))])
def test_load_weights_h5_incompatible_shapes(tmp_path, shape1, shape2):
    parameters = create_model_zoo_parameters(tmp_path, shape1)

    with cwd(tmp_path):
        build_modelzoo(*parameters)

    # create a new model and load from previous weights
    model = create_model(tmp_path, shape2)

    # set previous weights
    with pytest.raises(ValueError):
        load_weights(model, str(parameters[0]))
