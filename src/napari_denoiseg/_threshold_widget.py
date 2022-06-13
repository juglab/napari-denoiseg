from pathlib import Path

import bioimageio.core
import napari
from napari.qt.threading import thread_worker
import numpy as np
from napari_denoiseg._train_widget import State, generate_config, create_choice_widget
from napari_denoiseg._predict_widget import get_load_button
from napari_denoiseg._folder_widget import FolderWidget

import numpy as np
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
    QLabel,
    QTabWidget
)


class ThresholdWiget(QWidget):
    def __init__(self, napari_viewer):
        super().__init__()

        self.state = State.IDLE
        self.viewer = napari_viewer

        self.setLayout(QVBoxLayout())
        self.setMaximumHeight(300)

        ###############################
        # QTabs
        self.tabs = QTabWidget()
        tab_layers = QWidget()
        tab_layers.setLayout(QVBoxLayout())

        tab_disk = QWidget()
        tab_disk.setLayout(QVBoxLayout())

        # add tabs
        self.tabs.addTab(tab_layers, 'From layers')
        self.tabs.addTab(tab_disk, 'From disk')
        self.tabs.setMaximumHeight(150)

        # layer tabs
        self.layer_choice = create_choice_widget(napari_viewer)
        self.images = self.layer_choice.Images
        self.labels = self.layer_choice.Masks
        tab_layers.layout().addWidget(self.layer_choice.native)

        # disk tab
        self.train_images_folder = FolderWidget('Choose')
        self.train_labels_folder = FolderWidget('Choose')

        buttons = QWidget()
        form = QFormLayout()

        form.addRow('Images', self.train_images_folder)
        form.addRow('Labels', self.train_labels_folder)

        buttons.setLayout(form)
        tab_disk.layout().addWidget(buttons)

        # add to main layout
        self.layout().addWidget(self.tabs)

        ###############################
        # others

        # load model button
        self.load_button = get_load_button()
        self.layout().addWidget(self.load_button.native)

        # predict button
        self.worker = None
        self.seg_prediction = None
        self.optimize_button = QPushButton("Optimize", self)
        self.optimize_button.clicked.connect(self.start_optimization)
        self.layout().addWidget(self.optimize_button)

        # feedback to users

    def start_optimization(self):
        if self.state == State.IDLE:
            self.state = State.RUNNING

            self.optimize_button.setText('Stop') # TODO make sure we actually interrupt the thread

            self.worker = optimizer_worker(self)
            self.worker.returned.connect(self.done)
            self.worker.start()
        elif self.state == State.RUNNING:
            self.state = State.IDLE

    def done(self):
        self.state = State.IDLE
        self.optimize_button.setText('Optimize')


@thread_worker(start_thread=False)
def optimizer_worker(widget: ThresholdWiget):
    from denoiseg.models import DenoiSeg
    from denoiseg.utils.compute_precision_threshold import measure_precision

    # images
    image_data = widget.images.value.data
    label_data = widget.labels.value.data
    assert image_data.shape == label_data.shape

    # instantiate model
    config = generate_config(image_data, 1, 1, 1)  # TODO what if model won't fit?
    basedir = 'models'
    weight_name = widget.load_button.Model.value
    name = weight_name.stem

    if widget.load_button.Model.value.suffix == ".zip":
        # we assume we got a modelzoo file
        rdf = bioimageio.core.load_resource_description(widget.load_button.Model.value)
        weight_name = rdf.weights['keras_hdf5'].source

    # create model
    model = DenoiSeg(config, name, basedir)
    model.keras_model.load_weights(weight_name)

    # threshold validation data to estimate the best threshold
    threshold, val_score = widget.model.optimize_thresholds(validation_x,
                                                            validation_y,
                                                            measure=measure_precision())



if __name__ == "__main__":
    from napari_denoiseg._sample_data import denoiseg_data_n0

    data = denoiseg_data_n0()

    # create a Viewer
    viewer = napari.Viewer()

    # add our plugin
    viewer.window.add_dock_widget(ThresholdWiget(viewer))

    # add images
    viewer.add_image(data[0][0][0:15], name=data[0][1]['name'])
    viewer.add_labels(data[1][0][0:15], name=data[1][1]['name'])

    napari.run()
