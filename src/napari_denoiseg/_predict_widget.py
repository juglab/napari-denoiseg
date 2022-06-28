"""

"""
import numpy as np
from qtpy.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QPushButton,
    QTabWidget,
    QProgressBar
)

import napari
from napari_denoiseg.utils import State, UpdateType
from napari_denoiseg.utils import FolderWidget
from napari_denoiseg.utils import layer_choice, load_button, threshold_spin
from napari_denoiseg.utils import prediction_worker
from napari_denoiseg.utils.widgets.magicgui_widgets import enable_3d

SEGMENTATION = 'segmented'
DENOISING = 'denoised'


class PredictWidget(QWidget):
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

        # image layer tab
        self.images = layer_choice(annotation=napari.layers.Image, name="Images")
        self.layout().addWidget(self.images.native)
        tab_layers.layout().addWidget(self.images.native)

        # disk tab
        self.images_folder = FolderWidget('Choose')
        tab_disk.layout().addWidget(self.images_folder)

        # add to main layout
        self.layout().addWidget(self.tabs)

        ###############################
        # others

        # load model button
        self.load_button = load_button()
        self.layout().addWidget(self.load_button.native)

        # load 3D enabling checkbox
        self.enable_3d = enable_3d()
        self.layout().addWidget(self.enable_3d.native)


        # threshold slider
        self.threshold_spin = threshold_spin()
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
        self.predict_button.clicked.connect(self.start_prediction)
        self.layout().addWidget(self.predict_button)

        self.n_im = 0
        self.load_from_disk = 0
        # napari_viewer.window.qt_viewer.destroyed.connect(self.interrupt)

    def update(self, updates):
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

    def interrupt(self):
        self.worker.quit()

    def start_prediction(self):
        if self.state == State.IDLE:
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
            self.seg_prediction = np.zeros(self.images.value.data.shape, dtype=np.int16)
            viewer.add_labels(self.seg_prediction, name=SEGMENTATION, opacity=0.5, visible=True)
            self.denoi_prediction = np.zeros(self.images.value.data.shape, dtype=np.int16)
            viewer.add_image(self.denoi_prediction, name=DENOISING, visible=True)

            # start the prediction worker
            self.worker = prediction_worker(self)
            self.worker.yielded.connect(lambda x: self.update(x))
            self.worker.returned.connect(self.done)
            self.worker.start()
        elif self.state == State.RUNNING:
            # stop requested
            self.state = State.IDLE

    def done(self):
        self.state = State.IDLE
        self.predict_button.setText('Predict again')


if __name__ == "__main__":
    from napari_denoiseg._sample_data import denoiseg_data_n0

    data = denoiseg_data_n0()

    # create a Viewer
    viewer = napari.Viewer()

    # add our plugin
    viewer.window.add_dock_widget(PredictWidget(viewer))

    # add images
    viewer.add_image(data[0][0][0:30], name=data[0][1]['name'])

    napari.run()
