"""
"""
from __future__ import annotations

import os
import urllib
import zipfile
import numpy as np

from napari.types import LayerDataTuple


def _download_data_2D(noise_level):
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


def _load_data_2D(noise_level):
    train_data = np.load('data/DSB2018_{}/train/train_data.npz'.format(noise_level))
    images = train_data['X_train'].astype(np.float32)
    labels = train_data['Y_train'].astype(np.int32)

    return [(images, {'name': 'Data'}), (labels, {"name": 'Labels'})]


def _denoiseg_data_2D(noise_level):
    # create a folder for the data
    if not os.path.isdir('./data'):
        os.mkdir('data')

    _download_data_2D(noise_level)

    return _load_data_2D(noise_level)


def _denoiseg_data_3D():
    # create a folder for our data
    if not os.path.isdir('./data'):
        os.mkdir('data')

    link = 'https://github.com/juglab/EmbedSeg/releases/download/v0.1.0/Mouse-Organoid-Cells-CBG.zip'

    # check if data has been downloaded already
    zipPath = 'data/Mouse-Organoid-Cells-CBG.zip'

    # download the zip archive
    if not os.path.exists(zipPath):
        urllib.request.urlretrieve(link, zipPath)

    # unzip the files
    if not os.path.exists(zipPath[:-4]):
        with zipfile.ZipFile(zipPath, 'r') as zip_ref:
            zip_ref.extractall('data')


def denoiseg_data_2D_n0() -> LayerDataTuple:
    return _denoiseg_data_2D('n0')


def denoiseg_data_2D_n10() -> LayerDataTuple:
    return _denoiseg_data_2D('n10')


def denoiseg_data_2D_n20() -> LayerDataTuple:
    return _denoiseg_data_2D('n20')


def denoiseg_data_3D() -> LayerDataTuple:
    return _denoiseg_data_3D()
