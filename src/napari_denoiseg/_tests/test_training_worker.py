import pytest
import numpy as np
from denoiseg.models import DenoiSeg

from napari_denoiseg.utils import generate_config
from napari_denoiseg.utils.training_worker import (
    sanity_check_validation_fraction,
    sanity_check_training_size,
    get_validation_patch_shape,
    normalize_images,
    prepare_data_disk,
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


@pytest.mark.parametrize('shape', [(16, 16), (16, 16, 8)])
def test_prepare_data_disk(tmp_path, shape):
    folders = ['train_x', 'train_y', 'val_x', 'val_y']
    sizes = [20, 5, 8, 8]

    # create data
    create_data(tmp_path, folders, sizes, shape)

    # load data
    X, Y, X_val, Y_val, x_val, y_val = prepare_data_disk(tmp_path / folders[0],
                                                         tmp_path / folders[1],
                                                         tmp_path / folders[2],
                                                         tmp_path / folders[3]
                                                         )

    assert X.shape[0] == sizes[0]*8  # augmentation
    assert X.shape[1:-1] == shape
    assert X.shape[-1] == 1
    assert Y.shape[0] == sizes[0]*8  # empty frames are added when there is no Y
    assert Y.shape[1:-1] == shape
    assert Y.shape[-1] == 3  # one hot-encoding, 3 classes

    assert X_val.shape[0] == sizes[2]  # no augmentation
    assert X_val.shape[1:-1] == shape
    assert X_val.shape[-1] == 1
    assert Y_val.shape[0] == sizes[3]
    assert Y_val.shape[1:-1] == shape
    assert Y_val.shape[-1] == 3  # one hot-encoding, 3 classes

    assert x_val.shape == X_val.shape[:-1]
    assert y_val.shape == Y_val.shape[:-1]


@pytest.mark.parametrize('shape', [(16, 16), (16, 16, 8)])
def test_prepare_data_disk_unpaired_val(tmp_path, shape):
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


@pytest.mark.parametrize('shape', [(8,), (8, 16, 16, 32), (4, 8, 16, 16, 32)])
def test_prepare_data_disk_wrong_dims(tmp_path, shape):
    folders = ['train_x', 'train_y', 'val_x', 'val_y']
    sizes = [20, 5, 8, 8]

    # create data
    create_data(tmp_path, folders, sizes, shape)

    # load data
    with pytest.raises(ValueError):
        prepare_data_disk(tmp_path / folders[0],
                          tmp_path / folders[1],
                          tmp_path / folders[2],
                          tmp_path / folders[3]
                          )


@pytest.mark.parametrize('shape', [(8,), (16, 16), (16, 16, 8), (32, 16, 16, 8), (32, 16, 16, 8, 3)])
def test_zero_sum





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

























