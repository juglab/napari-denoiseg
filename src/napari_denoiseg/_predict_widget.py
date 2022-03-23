"""
"""
from pathlib import Path
import napari
from napari.qt.threading import thread_worker
from magicgui import magic_factory
from magicgui.widgets import create_widget
import numpy as np
from ._train_widget import State
from qtpy.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QPushButton,
    QProgressBar
)
from enum import Enum


@magic_factory(auto_call=True,
               Threshold={"widget_type": "FloatSpinBox", "min": 0, "max": 1., "step": 0.1, 'value': 0.6})
def get_threshold_spin(Threshold: int):
    pass


@magic_factory(auto_call=True, Model={'mode': 'r', 'filter': '*.h5 *.bioimage.io.zip'})
def get_load_button(Model: Path):
    pass


def layer_choice_widget(np_viewer, annotation, **kwargs):
    widget = create_widget(annotation=annotation, **kwargs)
    widget.reset_choices()
    np_viewer.layers.events.inserted.connect(widget.reset_choices)
    np_viewer.layers.events.removed.connect(widget.reset_choices)
    return widget


class Updates(Enum):
    N_IMAGES = 'number of images'
    IMAGE = 'image'
    DONE = 'done'


class PredictWidget(QWidget):
    def __init__(self, napari_viewer):
        super().__init__()

        self.state = State.IDLE
        self.viewer = napari_viewer

        self.setLayout(QVBoxLayout())

        # load model button
        self.load_button = get_load_button()
        self.layout().addWidget(self.load_button.native)

        # image layer
        self.images = layer_choice_widget(napari_viewer, annotation=napari.layers.Image, name="Images")
        self.layout().addWidget(self.images.native)

        # threshold slider
        self.threshold_spin = get_threshold_spin()
        self.layout().addWidget(self.threshold_spin.native)

        # progress bar
        self.pb_prediction = QProgressBar()
        self.pb_prediction.setValue(0)
        self.pb_prediction.setMinimum(0)
        self.pb_prediction.setMaximum(100)
        self.pb_prediction.setTextVisible(True)
        self.pb_prediction.setFormat(f'Images ?/?')

        # predict button
        self.worker = None
        self.predict_button = QPushButton("Predict", self)
        self.predict_button.clicked.connect(self.start_prediction)
        self.layout().addWidget(self.predict_button)

        # this allows stopping the thread when the napari window is closed,
        # including reducing the risk that an update comes after closing the
        # window and appearing as a new Qt view. But the call to qt_viewer
        # will be deprecated. Hopefully until then an on_window_closing event
        # will be available.
        napari_viewer.window.qt_viewer.destroyed.connect(self.interrupt)

    def interrupt(self):
        self.worker.quit()

    def start_prediction(self):
        if self.state == State.IDLE:
            self.state = State.RUNNING

            self.predict_button.setText('Stop')

            self.worker = prediction_worker(self)
            self.worker.yielded.connect(lambda x: self.update_all(x))
            self.worker.returned.connect(self.done)
            self.worker.start()
        elif self.state == State.RUNNING:
            self.state = State.IDLE

    def done(self):
        self.state = State.IDLE
        self.train_button.setText('Predict again')


@thread_worker(start_thread=False)
def prediction_worker(widget: PredictWidget):

    # get number of images (2D slices)
    # yield total number of images
    # instantiate model
    # set weight using load
    # create label layer
    # loop over slices
        # yield image number + 1
        # predict
        # add prediction to layers
    # update done
    pass


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
