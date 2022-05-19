"""
"""
from pathlib import Path

import bioimageio.core
import napari
from napari.qt.threading import thread_worker
from magicgui import magic_factory
import numpy as np
from napari_denoiseg._train_widget import State, generate_config
from qtpy.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QPushButton,
    QProgressBar
)
from enum import Enum

RAW = 'raw'
SEGMENTATION = 'segmented'
DENOISING = 'denoised'


@magic_factory(auto_call=True,
               Threshold={"widget_type": "FloatSpinBox", "min": 0, "max": 1., "step": 0.1, 'value': 0.6})
def get_threshold_spin(Threshold: int):
    pass


@magic_factory(auto_call=True, Images={'mode': 'd'})
def get_load_button_imgs(Images: Path):
    pass


@magic_factory(auto_call=True, Model={'mode': 'r', 'filter': '*.h5 *.zip'})
def get_load_button_model(Model: Path):
    pass


class Updates(Enum):
    N_IMAGES = 'number of images'
    IMAGE = 'image'
    DONE = 'done'


class BatchPredictWidget(QWidget):
    def __init__(self, napari_viewer):
        super().__init__()

        self.state = State.IDLE
        self.viewer = napari_viewer

        self.setLayout(QVBoxLayout())

        # load model button
        self.load_model_button = get_load_button_model()
        self.layout().addWidget(self.load_model_button.native)

        # load image
        self.load_img_button = get_load_button_imgs()
        self.layout().addWidget(self.load_img_button.native)

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
        self.layout().addWidget(self.pb_prediction)

        # predict button
        self.images = None
        self.worker = None
        self.seg_prediction = None
        self.denoi_prediction = None
        self.predict_button = QPushButton("Predict", self)
        self.predict_button.clicked.connect(self.start_prediction)
        self.layout().addWidget(self.predict_button)

        self.n_im = 0

        # napari_viewer.window.qt_viewer.destroyed.connect(self.interrupt)

    def update(self, updates):
        if Updates.N_IMAGES in updates:
            self.n_im = updates[Updates.N_IMAGES]
            self.pb_prediction.setValue(0)
            self.pb_prediction.setFormat(f'Prediction 0/{self.n_im}')

        if Updates.IMAGE in updates:
            val = updates[Updates.IMAGE]
            perc = int(100 * val / self.n_im + 0.5)
            self.pb_prediction.setValue(perc)
            self.pb_prediction.setFormat(f'Prediction {val}/{self.n_im}')
            self.viewer.layers[SEGMENTATION].refresh()

        if Updates.DONE in updates:
            self.pb_prediction.setValue(100)
            self.pb_prediction.setFormat(f'Prediction done')

    def interrupt(self):
        self.worker.quit()

    def start_prediction(self):
        if self.state == State.IDLE:
            import os
            from PIL import Image

            self.state = State.RUNNING

            self.predict_button.setText('Stop')

            if SEGMENTATION in self.viewer.layers:
                self.viewer.layers.remove(SEGMENTATION)
            if DENOISING in self.viewer.layers:
                self.viewer.layers.remove(DENOISING)

            # load images # TODO: wouldn't it be better to have it in the predict thread?
            img_paths = self.load_img_button.Images.value
            imgs = []
            for f in os.listdir(img_paths):
                im = Image.open(os.path.join(img_paths, f))
                imgs.append(np.array(im))
            self.images = np.array(imgs)

            viewer.add_image(self.images, name=RAW, visible=True)
            self.seg_prediction = np.zeros(self.images.shape, dtype=np.int16)
            viewer.add_labels(self.seg_prediction, name=SEGMENTATION, opacity=0.5, visible=True)
            self.denoi_prediction = np.zeros(self.images.shape, dtype=np.int16)
            viewer.add_image(self.denoi_prediction, name=DENOISING, visible=True)

            self.worker = prediction_worker(self)
            self.worker.yielded.connect(lambda x: self.update(x))
            self.worker.returned.connect(self.done)
            self.worker.start()
        elif self.state == State.RUNNING:
            self.state = State.IDLE

    def done(self):
        self.state = State.IDLE
        self.predict_button.setText('Predict again')


@thread_worker(start_thread=False)
def prediction_worker(widget: BatchPredictWidget):
    from denoiseg.models import DenoiSeg
    import tensorflow as tf

    X = widget.images[np.newaxis, 0, :, :, np.newaxis]
    print(X.shape)

    # yield total number of images
    n_img = widget.images.shape[0]  # this will break down
    yield {Updates.N_IMAGES: n_img}

    # instantiate model
    config = generate_config(X, 1, 1, 1)  # here no way to tell if the network size corresponds to the one saved...
    basedir = 'models'
    weight_name = widget.load_model_button.Model.value
    name = weight_name.stem

    if widget.load_model_button.Model.value.suffix == ".zip":
        # we assume we got a modelzoo file
        rdf = bioimageio.core.load_resource_description(widget.load_model_button.Model.value)
        weight_name = rdf.weights['keras_hdf5'].source

    # this is to prevent the memory from saturating on the gpu on my machine
    if tf.config.list_physical_devices('GPU'):
        tf.config.experimental.set_memory_growth(tf.config.list_physical_devices('GPU')[0], True)
    model = DenoiSeg(config, name, basedir)

    # set weight using load
    model.keras_model.load_weights(weight_name)

    # loop over slices
    for i in range(widget.images.shape[0]):
        # yield image number + 1
        yield {Updates.IMAGE: i + 1}

        # predict
        pred = model.predict(widget.images[np.newaxis, i, :, :, np.newaxis], axes='SYXC')

        # threshold
        pred_seg = pred[0, :, :, 2] >= widget.threshold_spin.Threshold.value

        # add prediction to layers
        widget.seg_prediction[i, :, :] = pred_seg
        widget.denoi_prediction[i, :, :] = pred[0, :, :, 0]

        # check if stop requested
        if widget.state != State.RUNNING:
            break

    # update done
    yield {Updates.DONE}


if __name__ == "__main__":
    # create a Viewer
    viewer = napari.Viewer()

    # add our plugin
    viewer.window.add_dock_widget(BatchPredictWidget(viewer))

    napari.run()
