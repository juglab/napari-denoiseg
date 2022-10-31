"""
"""
from pathlib import Path
import urllib
import zipfile
from typing import Union

import numpy as np

from napari.types import LayerDataTuple

from napari_denoiseg.utils import cwd, get_default_path


def _download_data_3D(noise_level):
    assert noise_level in ['n10', 'n20']

    with cwd(get_default_path()):
        data_path = Path('data', 'Mouse-Organoid-Cells').absolute()
        if not data_path.exists():
            data_path.mkdir(parents=True)

        if noise_level == 'n10':
            link = 'https://download.fht.org/jug/denoiseg/Mouse-Organoid-Cells-CBG-128_n10.zip'
        elif noise_level == 'n20':
            link = 'https://download.fht.org/jug/denoiseg/Mouse-Organoid-Cells-CBG-128_n20.zip'

        # check if data has been downloaded already
        zip_name = 'Mouse-Organoid-Cells-CBG-128_{}.zip'.format(noise_level)
        zip_path = Path(data_path, zip_name)
        if not zip_path.exists():
            # download and unzip data
            urllib.request.urlretrieve(link, zip_path)
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(data_path)

        return Path(data_path, 'Mouse-Organoid-Cells-CBG-128')


def _load_data_3D(path: Union[str, Path], noise_level):
    name = 'train_data_{}.npz'.format(noise_level)
    data_path = Path(path, name)

    train_data = np.load(
        data_path,
        allow_pickle=True
    )
    X_train = train_data['X_train'].astype(np.float32)
    Y_train = train_data['Y_train'].astype(np.int32)

    return [(X_train, {'name': 'Train data'}),
            (Y_train, {'name': 'Train labels'})]


def _denoiseg_data_3D(noise_level):
    path = _download_data_3D(noise_level)

    return _load_data_3D(path, noise_level)


def _download_data_2D(noise_level):
    assert noise_level in ['n0', 'n10', 'n20']

    with cwd(get_default_path()):
        data_path = Path('data', 'DSB2018').absolute()
        if not data_path.exists():
            data_path.mkdir(parents=True)

        if noise_level == 'n0':
            link = 'https://zenodo.org/record/5156969/files/DSB2018_n0.zip?download=1'
        elif noise_level == 'n10':
            link = 'https://zenodo.org/record/5156977/files/DSB2018_n10.zip?download=1'
        elif noise_level == 'n20':
            link = 'https://zenodo.org/record/5156983/files/DSB2018_n20.zip?download=1'

        # check if data has been downloaded already
        zip_name = 'DSB2018_{}.zip'.format(noise_level)
        zip_path = Path(data_path, zip_name)
        if not zip_path.exists():
            # download and unzip data
            urllib.request.urlretrieve(link, zip_path)
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(data_path)

        return Path(data_path, zip_name[:-len('.zip')])


def _load_data_2D(path: Union[str, Path]):
    data_path = Path(path, 'train', 'train_data.npz')

    train_data = np.load(data_path, allow_pickle=True)
    X_train = train_data['X_train'].astype(np.float32)
    Y_train = train_data['Y_train'].astype(np.int32)

    return [(X_train, {'name': 'Train data'}),
            (Y_train[:50], {'name': 'Train labels'})]


def _denoiseg_data_2D(noise_level):
    path = _download_data_2D(noise_level)

    return _load_data_2D(path)


def demo_files():
    with cwd(get_default_path()):
        # load sem validation
        img = _denoiseg_data_2D('n10')[0][0][:50]

        # create models folder if it doesn't already exist
        model_path = Path('models', 'trained_DSB2018_n20').absolute()
        if not model_path.exists():
            model_path.mkdir(parents=True)

        # download sem model
        model_zip_path = Path(model_path, 'trained_DSB2018_n20.zip')
        if not model_zip_path.exists():
            # download and unzip data
            urllib.request.urlretrieve('https://download.fht.org/jug/napari/trained_DSB2018_n20.zip', model_zip_path)
            with zipfile.ZipFile(model_zip_path, 'r') as zip_ref:
                zip_ref.extractall(model_path)

        return img, Path(model_path, 'DSB2018_n20.h5')


def denoiseg_data_2D_n0() -> LayerDataTuple:
    return _denoiseg_data_2D('n0')


def denoiseg_data_2D_n10() -> LayerDataTuple:
    return _denoiseg_data_2D('n10')


def denoiseg_data_2D_n20() -> LayerDataTuple:
    return _denoiseg_data_2D('n20')


def denoiseg_data_3D_n10() -> LayerDataTuple:
    return _denoiseg_data_3D('n10')


def denoiseg_data_3D_n20() -> LayerDataTuple:
    return _denoiseg_data_3D('n20')
