import urllib
import zipfile

import napari
import os

import numpy as np

from src.napari_denoiseg import CustomWidget, denoiseg_widget

with napari.gui_qt():
    # create a folder for our data
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

    # Loading of the training images
    train_data = np.load('data/DSB2018_{}/train/train_data.npz'.format(noise_level))
    images = train_data['X_train'].astype(np.float32)[0:30, :, :]
    labels = train_data['Y_train'].astype(np.int32)[0:16, :, :]

    # create a Viewer and add an image here
    viewer = napari.Viewer()

    # add images
    viewer.add_image(images)
    viewer.add_labels(labels)

    # custom code to add data here
    viewer.window.add_dock_widget(denoiseg_widget())

    napari.run()
