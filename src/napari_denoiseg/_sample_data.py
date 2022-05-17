"""
This module is an example of a barebones sample data provider for napari.
It implements the "sample data" specification.
see: https://napari.org/plugins/stable/npe2_manifest_specification.html
Replace code below according to your needs.
"""
from __future__ import annotations

import os
import urllib
import zipfile
import numpy as np

from napari.types import LayerDataTuple


def denoiseg_data_n0() -> LayerDataTuple:
    # create a folder for the data
    if not os.path.isdir('./data'):
        os.mkdir('data')

    noise_level = 'n0'
    link = 'https://zenodo.org/record/5156969/files/DSB2018_n0.zip?download=1'

    # check if data has been downloaded already
    zipPath = "data/DSB2018_{}.zip".format(noise_level)
    if not os.path.exists(zipPath):
        # download and unzip data
        data = urllib.request.urlretrieve(link, zipPath)
        with zipfile.ZipFile(zipPath, 'r') as zip_ref:
            zip_ref.extractall("data")

    train_data = np.load('data/DSB2018_{}/train/train_data.npz'.format(noise_level))
    images = train_data['X_train'].astype(np.float32)
    labels = train_data['Y_train'].astype(np.int32)

    return [(images, {'name': 'Data'}), (labels, {"name": 'Labels'})]