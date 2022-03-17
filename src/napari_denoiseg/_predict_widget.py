"""
"""
import numpy as np
import napari
from qtpy.QtWidgets import (
    QWidget
)


class PredictWidget(QWidget):
    def __init__(self, napari_viewer):
        super().__init__()



if __name__ == "__main__":
    with napari.gui_qt():
        noise_level = 'n0'

        # Loading of the training images
        train_data = np.load('data/DSB2018_{}/train/train_data.npz'.format(noise_level))
        images = train_data['X_train'].astype(np.float32)[0:30, :, :]

        # create a Viewer and add an image here
        viewer = napari.Viewer()

        # custom code to add data here
        viewer.window.add_dock_widget(PredictWidget(viewer))

        # add images
        viewer.add_image(images, name='Images')

        napari.run()
