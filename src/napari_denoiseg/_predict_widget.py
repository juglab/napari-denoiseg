"""
"""
import napari
from napari.qt.threading import thread_worker
from magicgui import magic_factory
from magicgui.widgets import create_widget, FloatSpinBox
from queue import Queue
import numpy as np
from utils import TBPlotWidget
from _train_widget import State
from qtpy.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QPushButton,
    QProgressBar,
    QSpinBox,
    QFormLayout,
    QComboBox,
    QFileDialog,
    QLabel
)
from enum import Enum


@magic_factory(auto_call=True,
               labels=False,
               slider={"widget_type": "FloatSpinBox", "min": 0, "max": 1., "step": 0.1, 'value': 0.6})
def get_threshold_spin(slider: int):
    pass


def layer_choice_widget(np_viewer, annotation, **kwargs):
    widget = create_widget(annotation=annotation, **kwargs)
    widget.reset_choices()
    np_viewer.layers.events.inserted.connect(widget.reset_choices)
    np_viewer.layers.events.removed.connect(widget.reset_choices)
    return widget


class Updates(Enum):
    IMAGE = 'image'
    DONE = 'done'


class Updater():
    def __init__(self):
        self.queue = Queue(10)
        self.image = 0

    def on_predict_begin(self, image):
        self.image = image
        self.queue.put({Updates.IMAGE: self.image + 1})

    def on_predict_end(self):
        self.queue.put(Updates.DONE)

    def stop_prediction(self):
        pass


class PredictWidget(QWidget):
    def __init__(self, napari_viewer):
        super().__init__()

        self.state = State.IDLE
        self.viewer = napari_viewer

        self.setLayout(QVBoxLayout())

        # load model button
        self.load_button = QPushButton("Load model", self)
        self.load_button.clicked.connect(self.load_model)
        self.layout().addWidget(self.load_button)

        # image layer
        self.images = layer_choice_widget(napari_viewer, annotation=napari.layers.Image, name="Images")
        self.layout().addWidget(self.images.native)

        # threshold slider
        self.threshold_spin = get_threshold_spin()
        others = QWidget()
        formLayout = QFormLayout()
        formLayout.addRow('Threshold', self.threshold_spin.native)
        others.setLayout(formLayout)
        self.layout().addWidget(others)

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

    def load_model(self):
        if self.state == State.IDLE:
            where = QFileDialog.getSaveFileName(caption='Load model', filter="Models (*.h5 *.bioimage.io.zip)")[0]
            print(where)


@thread_worker(start_thread=False)
def prediction_worker(widget: PredictWidget):
    import threading
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
