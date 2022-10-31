import numpy as np
import pytest

from napari_denoiseg._tests.test_utils import create_simple_model, save_img, save_weights_h5
from napari_denoiseg.utils.prediction_worker import _run_lazy_prediction, _run_prediction, _run_prediction_to_disk
from napari_denoiseg.utils import State, UpdateType, lazy_load_generator, load_from_disk


class MonkeyPatchWidget:
    def __init__(self, path):
        self.path = path
        self.state = State.RUNNING
        self.seg_prediction = None
        self.denoi_prediction = None

    def get_model_path(self):
        return self.path


@pytest.mark.parametrize('n', [2])
@pytest.mark.parametrize('t', [0.6])
@pytest.mark.parametrize('n_tiles', [2])
@pytest.mark.parametrize('shape, shape_denoiseg, axes',
                         [((16, 16), (1, 16, 16, 1), 'YX'),
                          ((2, 16, 16), (2, 16, 16, 1), 'TYX'),
                          ((16, 16, 16), (1, 16, 16, 16, 1), 'ZYX'),
                          ((2, 16, 16, 3), (2, 16, 16, 3), 'TYXC'),
                          ((16, 16, 16, 3), (1, 16, 16, 16, 3), 'ZYXC'),
                          ((2, 16, 16, 16, 3), (2, 16, 16, 16, 3), 'SZYXC')])
def test_run_lazy_prediction_same_size(tmp_path, t, n, n_tiles, shape, shape_denoiseg, axes):
    # create model and save it to disk
    model = create_simple_model(tmp_path, shape_denoiseg)
    path_to_h5 = save_weights_h5(model, tmp_path)

    # create files
    save_img(tmp_path, n, shape)

    # instantiate generator
    gen, m = lazy_load_generator(tmp_path)
    assert m == n

    # run prediction (it is a generator)
    mk = MonkeyPatchWidget(path_to_h5)
    parameters = (mk, model, axes, gen)
    hist = list(_run_lazy_prediction(*parameters,
                                     is_threshold=t != 0,  # threshold if t != 0
                                     threshold=t,
                                     is_tiled=n_tiles != 1,  # use tiles if n_tiles != 1
                                     n_tiles=n_tiles))
    assert hist[-1] == {UpdateType.DONE}
    assert len(hist) == n + 1

    # check that images have been saved
    folder = tmp_path / 'results'
    image_files = [f for f in folder.glob('*.tif*')]
    assert len(image_files) == 2 * n


@pytest.mark.parametrize('t', [0.6])
@pytest.mark.parametrize('n_tiles', [2])
@pytest.mark.parametrize('shape1, shape2, shape_denoiseg, axes',
                         [((16, 16), (32, 32), (1, 16, 16, 1), 'YX'),
                          ((2, 16, 16), (3, 32, 32), (2, 16, 16, 1), 'TYX'),
                          ((16, 32, 32), (32, 16, 16), (1, 16, 32, 32, 1), 'ZYX'),
                          ((5, 16, 16, 3), (3, 32, 32, 3), (5, 16, 16, 3), 'TYXC'),
                          ((16, 32, 32, 3), (32, 16, 16, 3), (1, 16, 32, 32, 3), 'ZYXC'),
                          ((5, 16, 32, 32, 3), (3, 16, 16, 16, 3), (5, 16, 32, 32, 3), 'SZYXC')])
def test_run_lazy_prediction_different_sizes(tmp_path, t, n_tiles, shape1, shape2, shape_denoiseg, axes):
    # create model and save it to disk
    model = create_simple_model(tmp_path, shape_denoiseg)
    path_to_h5 = save_weights_h5(model, tmp_path)

    # create files
    n = 1
    save_img(tmp_path, n, shape1, prefix='i1_')
    save_img(tmp_path, n, shape2, prefix='i2_')

    # instantiate generator
    gen, m = lazy_load_generator(tmp_path)
    assert m == 2 * n

    # run prediction (it is a generator)
    mk = MonkeyPatchWidget(path_to_h5)
    parameters = (mk, model, axes, gen)
    hist = list(_run_lazy_prediction(*parameters,
                                     is_threshold=t != 0,  # threshold if t != 0
                                     threshold=t,
                                     is_tiled=n_tiles != 1,  # use tiles if n_tiles != 1
                                     n_tiles=n_tiles))
    assert hist[-1] == {UpdateType.DONE}
    assert len(hist) == 2 * n + 1

    # check that images have been saved
    folder = tmp_path / 'results'
    image_files = [f for f in folder.glob('*.tif*')]
    assert len(image_files) == 2 * 2 * n


@pytest.mark.parametrize('t', [0.6])
@pytest.mark.parametrize('n_tiles', [2])
@pytest.mark.parametrize('shape1, shape2, shape_denoiseg, axes',
                         [((16, 16), (32, 32), (1, 16, 16, 1), 'YX'),
                          ((2, 16, 16), (3, 32, 32), (2, 16, 16, 1), 'TYX'),
                          ((16, 32, 32), (32, 16, 16), (1, 16, 32, 32, 1), 'ZYX'),
                          ((5, 16, 16, 3), (3, 32, 32, 3), (5, 16, 16, 3), 'TYXC'),
                          ((16, 32, 32, 3), (32, 16, 16, 3), (1, 16, 32, 32, 3), 'ZYXC'),
                          ((5, 16, 32, 32, 3), (3, 16, 16, 16, 3), (5, 16, 32, 32, 3), 'SZYXC')])
def test_run_from_disk_prediction_different_sizes(tmp_path, t, n_tiles, shape1, shape2, shape_denoiseg, axes):
    # create model and save it to disk
    model = create_simple_model(tmp_path, shape_denoiseg)
    path_to_h5 = save_weights_h5(model, tmp_path)

    # create files
    n = 1
    save_img(tmp_path, n, shape1, prefix='i1_')
    save_img(tmp_path, n, shape2, prefix='i2_')

    # load images
    images, new_axes = load_from_disk(tmp_path, axes)
    assert type(images) == tuple
    assert new_axes == axes if 'S' in axes else 'S' + axes

    # run prediction
    mk = MonkeyPatchWidget(path_to_h5)
    parameters = (mk, model, new_axes, images)
    hist = list(_run_prediction_to_disk(*parameters,
                                        is_threshold=t != 0,  # threshold if t != 0
                                        threshold=t,
                                        is_tiled=n_tiles != 1,  # use tiles if n_tiles != 1
                                        n_tiles=n_tiles))

    assert hist[-1] == {UpdateType.DONE}
    assert len(hist) == n * 2 + 2

    # check that images have been saved
    folder = tmp_path / 'results'
    image_files = [f for f in folder.glob('*.tif*')]
    assert len(image_files) == 2 * 2 * n


@pytest.mark.parametrize('n', [2])
@pytest.mark.parametrize('t', [0.6])
@pytest.mark.parametrize('n_tiles', [2])
@pytest.mark.parametrize('shape, shape_denoiseg, axes',
                         [((16, 16), (1, 16, 16, 1), 'YX'),
                          ((2, 16, 16), (2, 16, 16, 1), 'TYX'),
                          ((16, 16, 16), (1, 16, 16, 16, 1), 'ZYX'),
                          ((2, 16, 16, 3), (2, 16, 16, 3), 'TYXC'),
                          ((16, 16, 16, 3), (1, 16, 16, 16, 3), 'ZYXC'),
                          ((2, 16, 16, 16, 3), (2, 16, 16, 16, 3), 'SZYXC')])
def test_run_prediction_from_disk_numpy(tmp_path, n, t, n_tiles, shape, shape_denoiseg, axes):
    # create model and save it to disk
    model = create_simple_model(tmp_path, shape_denoiseg)
    path_to_h5 = save_weights_h5(model, tmp_path)

    # create files
    save_img(tmp_path, n, shape)

    # load images
    images, new_axes = load_from_disk(tmp_path, axes)

    # run prediction (it is a generator)
    mk = MonkeyPatchWidget(path_to_h5)
    parameters = (mk, model, new_axes, images)
    hist = list(_run_prediction(*parameters,
                                is_threshold=t != 0,  # threshold if t != 0
                                threshold=t,
                                is_tiled=n_tiles != 1,  # use tiles if n_tiles != 1
                                n_tiles=n_tiles))
    assert hist[-1] == {UpdateType.DONE}
    assert len(hist) == n * shape_denoiseg[0] + 2


@pytest.mark.qt
@pytest.mark.parametrize('t', [0.6])
@pytest.mark.parametrize('n_tiles', [2])
@pytest.mark.parametrize('shape, shape_denoiseg, axes',
                         [((16, 16), (1, 16, 16, 1), 'YX'),
                          ((2, 16, 16), (2, 16, 16, 1), 'TYX'),
                          ((16, 16, 16), (1, 16, 16, 16, 1), 'ZYX'),
                          ((2, 16, 16, 3), (2, 16, 16, 3), 'TYXC'),
                          ((16, 16, 16, 3), (1, 16, 16, 16, 3), 'ZYXC'),
                          ((2, 16, 16, 16, 3), (2, 16, 16, 16, 3), 'SZYXC')])
def test_run_prediction_from_layers(tmp_path, make_napari_viewer, t, n_tiles, shape, shape_denoiseg, axes):
    viewer = make_napari_viewer()

    # create images
    name = 'images'
    img = np.zeros(shape)
    viewer.add_image(img, name=name)

    # create model and save it to disk
    model = create_simple_model(tmp_path, shape_denoiseg)
    path_to_h5 = save_weights_h5(model, tmp_path)

    # run prediction (it is a generator)
    mk = MonkeyPatchWidget(path_to_h5)
    parameters = (mk, model, axes, viewer.layers['images'].data)
    hist = list(_run_prediction(*parameters,
                                is_threshold=t != 0,  # threshold if t != 0
                                threshold=t,
                                is_tiled=n_tiles != 1,  # use tiles if n_tiles != 1
                                n_tiles=n_tiles))
    assert hist[-1] == {UpdateType.DONE}
    assert len(hist) == shape_denoiseg[0] + 2
