import os
from pathlib import Path
import numpy as np
from tifffile import imwrite

from src.napari_denoiseg.utils.raw_data_loader import from_folder


def save_img(folder_path, n, shape):
    for i in range(n):
        im = np.random.randint(0, 255, shape, dtype=np.uint16)
        imwrite(os.path.join(folder_path, str(i) + '.tif'), im)


def create_data(dir, folders, sizes, shape):
    for n, f in zip(sizes, folders):
        source = dir / f
        os.mkdir(source)
        save_img(source, n, shape)


def test_create_data(tmpdir):
    folders = ['train_X', 'train_Y', 'val_X', 'val_Y']
    sizes = [20, 8, 5, 5]

    create_data(tmpdir, folders, sizes, (1, 8, 16, 16))
    # lo

    for n, f in zip(sizes, folders):
        source = tmpdir / f
        files = [f for f in Path(source).glob('*.tif*')]
        assert len(files) == n


def test_raw_data_loader_unequal_sizes(tmpdir):
    folders = ['train_X', 'train_Y']
    sizes = [15, 5]

    create_data(tmpdir, folders, sizes, (3, 16, 16))

    data = from_folder(tmpdir, folders[0], folders[1], check_exists=False)

    n = 0
    n_empty = 0
    for source_x, target_y, _, _ in data.generator():
        n += 1

        if target_y.min() == target_y.max() == 0:
            n_empty += 1

    assert n == sizes[0]
    assert n_empty == sizes[0]-sizes[1]


def test_raw_data_loader_equal_sizes(tmpdir):
    folders = ['val_X', 'val_Y']
    sizes = [5, 5]

    create_data(tmpdir, folders, sizes, (3, 16, 16))

    data = from_folder(tmpdir, folders[0], folders[1], check_exists=False)

    n = 0
    n_empty = 0
    for source_x, target_y, _, _ in data.generator():
        n += 1

        if target_y.min() == target_y.max() == 0:
            n_empty += 1

    assert n == sizes[0]
    assert n_empty == sizes[0]-sizes[1]
