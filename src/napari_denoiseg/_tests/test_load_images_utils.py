import os

import pytest
import numpy as np

from napari_denoiseg._tests.test_utils import (
    save_img,
    create_data
)
from napari_denoiseg.utils import (
    load_pairs_generator,
    load_from_disk,
    load_pairs_from_disk,
    lazy_load_generator
)


# TODO: test when no image is found
# todo test when there is only a single image on the disk
# TODO write goal of each test
# TODO Make sure they are compatible with Y with C channel

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
# test load_from_disk
@pytest.mark.parametrize('shape, axes', [((8, 8), 'YX'),
                                         ((4, 8, 8), 'ZYX'),
                                         ((5, 8, 8), 'SYX'),
                                         ((5, 8, 8, 3), 'SYXT')])
def test_load_from_disk_same_shapes(tmp_path, shape, axes):
    n = 10
    save_img(tmp_path, n, shape)

    # load images
    images, new_axes = load_from_disk(tmp_path, axes)
    assert type(images) == np.ndarray

    if 'S' in axes:
        assert new_axes == axes
        assert len(images.shape) == len(shape)
        assert images.shape[0] == n * shape[0]
        assert images.shape[1:] == shape[1:]
    else:
        assert new_axes == 'S' + axes
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
    images, new_axes = load_from_disk(tmp_path, axes)
    assert type(images) == tuple
    assert len(images[0]) == n[0] + n[1]
    assert len(images[1]) == n[0] + n[1]
    assert new_axes == axes

    for img in images[0]:
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
    save_img(tmp_path / folders[1], n[1], shape)

    # load images, replacing n[1]-n[0] last images with blank frames
    X, Y, n_loaded = load_pairs_from_disk(tmp_path / folders[0], tmp_path / folders[1], axes, check_exists=False)
    assert Y[n[1]:, ...].min() == Y[n[1]:, ...].max() == 0
    assert n[0] == n_loaded

    # load images with the check_exists flag triggers error
    with pytest.raises(FileNotFoundError):
        load_pairs_from_disk(tmp_path / folders[0], tmp_path / folders[1], axes, check_exists=True)


@pytest.mark.parametrize('shape1, shape2, axes', [((8, 8), (16, 16), 'YX'),
                                                  ((4, 8, 8), (8, 16, 16), 'ZYX'),
                                                  ((16, 8, 32, 3), (12, 16, 32, 3), 'ZYXC'),
                                                  ((16, 8, 16, 32, 3), (10, 16, 32, 32, 3), 'TZYXC')])
def test_load_pairs_from_disk_different_shapes(tmp_path, shape1, shape2, axes):
    n = [5, 2]
    folders = ['X', 'Y']
    for f in folders:
        os.mkdir(tmp_path / f)

    save_img(tmp_path / folders[0], n[0], shape1, prefix='1_')
    save_img(tmp_path / folders[0], n[0], shape2, prefix='2_')

    if 'C' in axes:
        save_img(tmp_path / folders[1], n[1], shape1[:-1])
        save_img(tmp_path / folders[1], n[1], shape1[:-1])
    else:
        save_img(tmp_path / folders[1], n[1], shape1, prefix='1_')
        save_img(tmp_path / folders[1], n[1], shape2, prefix='2_')

    # load images
    X, Y, n_loaded = load_pairs_from_disk(tmp_path / folders[0], tmp_path / folders[1], axes, check_exists=False)
    assert type(X) == list and type(Y) == list
    assert len(X) == len(Y) == 2 * n[0]

    # load images with the check_exists flag triggers error
    with pytest.raises(FileNotFoundError):
        load_pairs_from_disk(tmp_path / folders[0], tmp_path / folders[1], axes, check_exists=True)


@pytest.mark.parametrize('shape1, shape2', [((8, 8), (4, 8, 8)), ((4, 8, 8), (8, 8)), ((8, 8), (8, 9))])
def test_load_pairs_from_disk_different_shape_sizes(tmp_path, shape1, shape2):
    n = 10
    folders = ['X', 'Y']
    for f in folders:
        os.mkdir(tmp_path / f)

    save_img(tmp_path / folders[0], n, shape1)
    save_img(tmp_path / folders[1], n, shape2)

    # load images
    with pytest.raises(ValueError):
        load_pairs_from_disk(tmp_path / folders[0], tmp_path / folders[1], 'YX')


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
