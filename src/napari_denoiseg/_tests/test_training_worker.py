import pytest
import numpy as np
from denoiseg.models import DenoiSeg

from napari_denoiseg.utils import generate_config
from napari_denoiseg.utils.training_worker import (
    sanity_check_validation_fraction,
    sanity_check_training_size,
    get_validation_patch_shape,
    normalize_images
)
from napari_denoiseg._tests.test_utils import (
    create_model
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
                                         ((1, 32, 64, 1), 'SXY'),
                                         ((1, 16, 16, 16, 1), 'SZXY'),
                                         ((1, 32, 128, 64, 1), 'SZXY')])
def test_sanity_check_training_size(tmp_path, shape, axes):
    model = create_model(tmp_path, shape)

    # run sanity check on training size
    sanity_check_training_size(np.zeros(shape), model, axes)


@pytest.mark.parametrize('shape, axes', [((1, 8, 16, 1), 'SXYC'),
                                         ((1, 16, 66, 1), 'SXYC'),
                                         ((1, 18, 64, 1), 'SXYC'),
                                         ((1, 16, 16, 8, 1), 'SZXYC'),
                                         ((1, 66, 128, 64, 1), 'SZXYC')])
def test_sanity_check_training_size(tmp_path, shape, axes):
    model = create_model(tmp_path, shape)

    # run sanity check on training size
    with pytest.raises(ValueError):
        sanity_check_training_size(np.zeros(shape), model, axes)


@pytest.mark.parametrize('shape_in, shape_out, axes', [((32, 8, 16, 1), (8, 16), 'SXYC'),
                                                       ((64, 8, 16, 32, 1), (16, 32, 8), 'SZXYC')])
def test_get_validation_patch_shape(shape_in, shape_out, axes):
    X_val = np.zeros(shape_in)

    assert get_validation_patch_shape(X_val, axes) == shape_out


@pytest.mark.parametrize('shape_train, shape_val', [((100000, 4, 4, 1), (100000, 4, 4, 1))])
def test_normalize_images(tmp_path, shape_train, shape_val):
    # create data
    np.random.seed(42)
    X_train = np.random.normal(10, 5, shape_train)
    X_val = np.random.normal(10, 5, shape_val)

    # create model
    name = 'myModel'
    config = generate_config(X_train)
    model = DenoiSeg(config, name, tmp_path)

    # normalize data
    X_train_norm, X_val_norm = normalize_images(model, X_train, X_val)
    assert (np.abs(np.mean(X_train_norm, axis=0)) < 0.01).all()
    assert (np.abs(np.std(X_train_norm, axis=0) - 1) < 0.01).all()
    assert (np.abs(np.mean(X_val_norm, axis=0)) < 0.01).all()
    assert (np.abs(np.std(X_val_norm, axis=0) - 1) < 0.01).all()

