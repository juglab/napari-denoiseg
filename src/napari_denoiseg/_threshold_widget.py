from pathlib import Path

import bioimageio.core
import napari
from napari.qt.threading import thread_worker
from napari_denoiseg._train_widget import State, generate_config, create_choice_widget
from napari_denoiseg._predict_widget import get_load_button
from napari_denoiseg._folder_widget import FolderWidget

import numpy as np
from qtpy.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QPushButton,
    QFormLayout,
    QTableWidget,
    QTabWidget,
    QTableWidgetItem,
    QHeaderView
)

T = 't'
M = 'metrics'


class ThresholdWiget(QWidget):
    def __init__(self, napari_viewer):
        super().__init__()

        self.state = State.IDLE
        self.viewer = napari_viewer

        self.setLayout(QVBoxLayout())
        self.setMaximumHeight(600)

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
        self.images_folder = FolderWidget('Choose')
        self.labels_folder = FolderWidget('Choose')

        buttons = QWidget()
        form = QFormLayout()

        form.addRow('Images', self.images_folder)
        form.addRow('Labels', self.labels_folder)

        buttons.setLayout(form)
        tab_disk.layout().addWidget(buttons)

        # add to main layout
        self.layout().addWidget(self.tabs)

        ###############################
        # others

        # load model button
        self.load_button = get_load_button()
        self.layout().addWidget(self.load_button.native)

        # feedback table to users
        self.table = QTableWidget()
        self.table.setRowCount(19)
        self.table.setColumnCount(2)

        self.table.setHorizontalHeaderLabels([T, M])
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.table.resizeRowsToContents()
        self.layout().addWidget(self.table)

        # predict button
        self.worker = None
        self.seg_prediction = None
        self.optimize_button = QPushButton("Optimize", self)
        self.optimize_button.clicked.connect(self.start_optimization)
        self.layout().addWidget(self.optimize_button)

        self.load_from_disk = 0
        self. results = None

    def start_optimization(self):
        if self.state == State.IDLE:
            self.state = State.RUNNING

            # register which data tab: layers or disk
            self.load_from_disk = self.tabs.currentIndex() == 1

            self.optimize_button.setText('Stop')
            self.table.clear()

            self.worker = optimizer_worker(self)
            self.worker.returned.connect(self.done)
            self.worker.yielded.connect(lambda x: self.update(x))
            self.worker.start()
        elif self.state == State.RUNNING:
            self.state = State.IDLE

    def update(self, score_tuple):
        (i, t, m) = score_tuple
        self.table.setItem(i, 0, QTableWidgetItem("{:.2f}".format(t)))
        self.table.setItem(i, 1, QTableWidgetItem("{:.2f}".format(m)))

    def done(self):
        self.state = State.IDLE
        self.optimize_button.setText('Optimize')


@thread_worker(start_thread=False)
def optimizer_worker(widget: ThresholdWiget):
    from denoiseg.models import DenoiSeg
    from denoiseg.utils.compute_precision_threshold import measure_precision

    # get images
    if widget.load_from_disk:
        from napari_denoiseg._raw_data_loader import from_folder

        images = Path(widget.images_folder.get_folder())
        labels = Path(widget.labels_folder.get_folder())

        # use generator to check whether pairs of similarly named images exist
        pairs = from_folder(images.parent, images.name, labels.name, axes='CYX')

        # load from disk
        _x_val = []
        _y_val = []
        for source_x, target_y, _, _ in pairs.generator():
            _x_val.append(source_x)
            _y_val.append(target_y)

        image_data, label_data = np.array(_x_val), np.array(_y_val, dtype=np.int)
    else:
        image_data = widget.images.value.data
        label_data = widget.labels.value.data
    assert image_data.shape == label_data.shape

    # instantiate model
    config = generate_config(image_data[np.newaxis, 0, ..., np.newaxis], 1, 1, 1)  # TODO what if model won't fit?
    basedir = 'models'

    weight_name = widget.load_button.Model.value
    assert len(weight_name.name) > 0
    name = weight_name.stem

    if widget.load_button.Model.value.suffix == ".zip":
        # we assume we got a modelzoo file
        rdf = bioimageio.core.load_resource_description(widget.load_button.Model.value)
        weight_name = rdf.weights['keras_hdf5'].source

    # create model
    model = DenoiSeg(config, name, basedir)
    model.keras_model.load_weights(weight_name)

    # threshold data to estimate the best threshold
    for i, ts in enumerate(np.linspace(0.1, 1, 19)):
        _, score = model.predict_label_masks(image_data, label_data, ts, measure_precision())

        # check if stop requested
        if widget.state != State.RUNNING:
            break

        yield i, ts, score


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
