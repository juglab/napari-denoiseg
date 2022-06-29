"""
"""
from __future__ import annotations

import os
import urllib
import zipfile
import numpy as np

from napari.types import LayerDataTuple


def _download_data(noise_level):
    assert noise_level in ['n0', 'n10', 'n20']

    if noise_level == 'n0':
        link = 'https://zenodo.org/record/5156969/files/DSB2018_n0.zip?download=1'
    elif noise_level == 'n10':
        link = 'https://zenodo.org/record/5156977/files/DSB2018_n10.zip?download=1'
    elif noise_level == 'n20':
        link = 'https://zenodo.org/record/5156983/files/DSB2018_n20.zip?download=1'

    # check if data has been downloaded already
    zipPath = "data/DSB2018_{}.zip".format(noise_level)
    if not os.path.exists(zipPath):
        # download and unzip data
        data = urllib.request.urlretrieve(link, zipPath)
        with zipfile.ZipFile(zipPath, 'r') as zip_ref:
            zip_ref.extractall("data")


def _load_data(noise_level):
    train_data = np.load('data/DSB2018_{}/train/train_data.npz'.format(noise_level))
    images = train_data['X_train'].astype(np.float32)
    labels = train_data['Y_train'].astype(np.int32)

    return [(images, {'name': 'Data'}), (labels, {"name": 'Labels'})]


def _denoiseg_data(noise_level):
    # create a folder for the data
    if not os.path.isdir('./data'):
        os.mkdir('data')

    _download_data(noise_level)

    return _load_data(noise_level)


def denoiseg_data_n0() -> LayerDataTuple:
    return _denoiseg_data('n0')


def denoiseg_data_n10() -> LayerDataTuple:
    return _denoiseg_data('n10')


def denoiseg_data_n20() -> LayerDataTuple:
    return _denoiseg_data('n20')
