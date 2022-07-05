"""

"""
from pathlib import Path

import numpy as np
from qtpy.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QPushButton,
    QTabWidget,
    QProgressBar,
    QCheckBox
)

import napari
from napari_denoiseg.utils import (
    State,
    UpdateType,
    prediction_worker,
    loading_worker
)
from napari_denoiseg.widgets import (
    FolderWidget,
    AxesWidget,
    layer_choice,
    load_button,
    threshold_spin
)

SEGMENTATION = 'segmented'
DENOISING = 'denoised'
SAMPLE = 'Sample data'


class PredictWidget(QWidget):
    def __init__(self, napari_viewer):
        super().__init__()

        self.state = State.IDLE
        self.viewer = napari_viewer

        self.setLayout(QVBoxLayout())
        self.setMaximumHeight(400)

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
        self.tabs.setMaximumHeight(250)

        # image layer tab
        self.images = layer_choice(annotation=napari.layers.Image, name="Images")
        tab_layers.layout().addWidget(self.images.native)

        # disk tab
        self.lazy_loading = QCheckBox('Lazy loading')
        tab_disk.layout().addWidget(self.lazy_loading)
        self.images_folder = FolderWidget('Choose')
        tab_disk.layout().addWidget(self.images_folder)

        # add to main layout
        self.layout().addWidget(self.tabs)
        self.images.choices = [x for x in napari_viewer.layers if type(x) is napari.layers.Image]

        ###############################
        # others

        # load model button
        self.load_model_button = load_button()
        self.layout().addWidget(self.load_model_button.native)

        # load 3D enabling checkbox
        self.enable_3d = QCheckBox('Enable 3D')
        self.layout().addWidget(self.enable_3d)

        # axes widget
        self.axes_widget = AxesWidget()
        self.layout().addWidget(self.axes_widget)

        # threshold slider
        self.threshold_cbox = QCheckBox('Apply threshold')
        self.layout().addWidget(self.threshold_cbox)
        self.threshold_spin = threshold_spin()
        self.threshold_spin.native.setEnabled(False)
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
        self.worker = None
        self.seg_prediction = None
        self.denoi_prediction = None
        self.predict_button = QPushButton("Predict", self)
        self.layout().addWidget(self.predict_button)

        # actions
        self.tabs.currentChanged.connect(self._update_tab_axes)
        self.predict_button.clicked.connect(self._start_prediction)
        self.images.changed.connect(self._update_layer_axes)
        self.images_folder.text_field.textChanged.connect(self._update_disk_axes)
        self.enable_3d.stateChanged.connect(self._update_3D)
        self.threshold_cbox.stateChanged.connect(self._update_threshold)

        # members
        self.n_im = 0
        self.shape = None
        self.load_from_disk = 0

        # update axes widget in case of data
        self._update_layer_axes()

    def _update_threshold(self):
        self.threshold_spin.native.setEnabled(self.threshold_cbox.isChecked())

    def _update_3D(self):
        self.axes_widget.update_is_3D(self.enable_3d.isChecked())
        self.axes_widget.set_text_field(self.axes_widget.get_default_text())

    def _update_layer_axes(self):
        if self.images.value is not None:
            self.shape = self.images.value.data.shape

            # update shape length in the axes widget
            self.axes_widget.update_axes_number(len(self.shape))
            self.axes_widget.set_text_field(self.axes_widget.get_default_text())

    def _add_image(self, image):
        if SAMPLE in self.viewer.layers:
            self.viewer.layers.remove(SAMPLE)

        if image is not None:
            self.viewer.add_image(image, name=SAMPLE, visible=True)
            self.shape = image.shape

            # update the axes widget
            self.axes_widget.update_axes_number(len(image.shape))
            self.axes_widget.set_text_field(self.axes_widget.get_default_text())

    def _update_disk_axes(self):
        path = self.images_folder.get_folder()

        # load one image
        load_worker = loading_worker(path)
        load_worker.yielded.connect(lambda x: self._add_image(x))
        load_worker.start()

    def _update_tab_axes(self):
        """
        Updates the axes widget following the newly selected tab.

        :return:
        """
        self.load_from_disk = self.tabs.currentIndex() == 1

        if self.load_from_disk:
            self._update_disk_axes()
        else:
            self._update_layer_axes()

    def _update(self, updates):
        """

        :param updates:
        :return:
        """
        if UpdateType.N_IMAGES in updates:
            self.n_im = updates[UpdateType.N_IMAGES]
            self.pb_prediction.setValue(0)
            self.pb_prediction.setFormat(f'Prediction 0/{self.n_im}')

        if UpdateType.IMAGE in updates:
            val = updates[UpdateType.IMAGE]
            perc = int(100 * val / self.n_im + 0.5)
            self.pb_prediction.setValue(perc)
            self.pb_prediction.setFormat(f'Prediction {val}/{self.n_im}')
            self.viewer.layers[SEGMENTATION].refresh()
            self.viewer.layers[DENOISING].refresh()

        if UpdateType.DONE in updates:
            self.pb_prediction.setValue(100)
            self.pb_prediction.setFormat(f'Prediction done')

    def _start_prediction(self):
        if self.state == State.IDLE:
            if self.axes_widget.is_valid() and Path(self.get_model_path()).exists():

                self.state = State.RUNNING

                self.predict_button.setText('Stop')

                # register which data tab: layers or disk
                self.load_from_disk = self.tabs.currentIndex() == 1

                # remove seg and denoising layers if they are present
                if SEGMENTATION in self.viewer.layers:
                    self.viewer.layers.remove(SEGMENTATION)
                if DENOISING in self.viewer.layers:
                    self.viewer.layers.remove(DENOISING)

                # create new seg and denoising layers
                # TODO 1: if self.threshold_cbox.isChecked() selected, seg_prediction is label, otherwise image layer
                # TODO 2: 1) find 'C' in self.get_axes,
                #         2) create (*shape[:ind_C], x, *shape[ind_C+1:]) from self.shape or images.value.data.shape,
                #          where x is 3 for the segmentation images
                #         3) denoised images has same dimensions than self.shape or images.value.data

                # axes in napari are anything ****YX which can be SZTCYX or CSTZYX etc.
                # but the axes widget tells us what the order is
                # DenoiSeg outputs S(Z)YXC, where T and S are fused (T * S)
                # In prediction worker (e.g. line 128), the shape must be the same than here

                if self.load_from_disk == 0:
                    self.seg_prediction = np.zeros(self.images.value.data.shape, dtype=np.float32)
                    viewer.add_labels(self.seg_prediction, name=SEGMENTATION, opacity=0.5, visible=True)
                    self.denoi_prediction = np.zeros(self.images.value.data.shape, dtype=np.float32)
                    viewer.add_image(self.denoi_prediction, name=DENOISING, visible=True)
                else:
                    self.seg_prediction = np.zeros(self.shape, dtype=np.float32)
                    viewer.add_labels(self.seg_prediction, name=SEGMENTATION, opacity=0.5, visible=True)
                    self.denoi_prediction = np.zeros(self.shape, dtype=np.float32)
                    viewer.add_image(self.denoi_prediction, name=DENOISING, visible=True)

                # start the prediction worker
                self.worker = prediction_worker(self)
                self.worker.yielded.connect(lambda x: self._update(x))
                self.worker.returned.connect(self._done)
                self.worker.start()
            else:
                # TODO feedback to users
                pass
        elif self.state == State.RUNNING:
            # stop requested
            self.state = State.IDLE

    def _done(self):
        self.state = State.IDLE
        self.predict_button.setText('Predict again')

    def get_model_path(self):
        return self.load_model_button.Model.value


if __name__ == "__main__":
    from napari_denoiseg._sample_data import denoiseg_data_2D_n0

    data = denoiseg_data_2D_n0()

    # create a Viewer
    viewer = napari.Viewer()

    # add our plugin
    viewer.window.add_dock_widget(PredictWidget(viewer))

    # add images
    viewer.add_image(data[0][0][0:30], name=data[0][1]['name'])

    napari.run()
