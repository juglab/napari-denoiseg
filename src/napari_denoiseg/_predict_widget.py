"""

"""
from pathlib import Path
from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QPushButton,
    QTabWidget,
    QProgressBar,
    QCheckBox,
    QGroupBox,
    QFormLayout
)

import napari
import napari.utils.notifications as ntf

from napari_denoiseg.utils import (
    State,
    UpdateType,
    prediction_worker,
    loading_worker
)
from napari_denoiseg.widgets import (
    FolderWidget,
    AxesWidget,
    BannerWidget,
    ScrollWidgetWrapper,
    create_gpu_label,
    layer_choice,
    load_button,
    threshold_spin,
    create_int_spinbox
)
from napari_denoiseg.resources import ICON_JUGLAB

SEGMENTATION = 'segmented'
DENOISING = 'denoised'
SAMPLE = 'Sample data'


class PredictWidgetWrapper(ScrollWidgetWrapper):
    def __init__(self, napari_viewer):
        self.widget = PredictWidget(napari_viewer)

        super().__init__(self.widget)


class PredictWidget(QWidget):
    def __init__(self, napari_viewer):
        super().__init__()

        self.state = State.IDLE
        self.viewer = napari_viewer

        self.setLayout(QVBoxLayout())

        # add banner
        self.layout().addWidget(BannerWidget('DenoiSeg - Prediction',
                                             ICON_JUGLAB,
                                             'A joint denoising and segmentation algorithm requiring '
                                             'only a few annotated ground truth images.',
                                             'https://juglab.github.io/napari_denoiseg',
                                             'https://github.com/juglab/napari_denoiseg/issues'))

        # add GPU button
        gpu_button = create_gpu_label()
        gpu_button.setAlignment(Qt.AlignmentFlag.AlignRight)
        self.layout().addWidget(gpu_button)

        # tabs
        self.tabs = QTabWidget()
        tab_layers = QWidget()
        tab_layers.setLayout(QVBoxLayout())

        tab_disk = QWidget()
        tab_disk.setLayout(QVBoxLayout())

        # add tabs
        self.tabs.addTab(tab_layers, 'From layers')
        self.tabs.addTab(tab_disk, 'From disk')
        self.tabs.setMaximumHeight(120)

        self.tabs.setTabToolTip(0, 'Use images from napari layers')
        self.tabs.setTabToolTip(1, 'Use images saved on the disk')

        # image layer tab
        self.images = layer_choice(annotation=napari.layers.Image, name="Images")
        tab_layers.layout().addWidget(self.images.native)
        self.images.native.setToolTip('Select an image layer')

        # disk tab
        self.lazy_loading = QCheckBox('Lazy loading')
        self.lazy_loading.setToolTip('Select in order to predict one image at a time')
        tab_disk.layout().addWidget(self.lazy_loading)

        self.images_folder = FolderWidget('Choose')
        tab_disk.layout().addWidget(self.images_folder)
        self.images_folder.setToolTip('Select a folder containing the images')

        # add to main layout
        self.layout().addWidget(self.tabs)
        self.images.choices = [x for x in napari_viewer.layers if type(x) is napari.layers.Image]

        # load model button
        self.loader_group = QGroupBox()
        self.loader_group.setMaximumHeight(400)
        self.loader_group.setTitle("Model Selection")
        self.loader_group.setLayout(QVBoxLayout())
        self.loader_group.layout().setContentsMargins(20, 20, 20, 0)

        self.load_model_button = load_button()
        self.load_model_button.native.setToolTip('Load a trained model (weights and configuration)')

        self.loader_group.layout().addWidget(self.load_model_button.native)
        self.layout().addWidget(self.loader_group)

        ###############################
        # parameters
        self.prediction_param_group = QGroupBox()
        self.prediction_param_group.setTitle("Parameters")
        self.prediction_param_group.setLayout(QVBoxLayout())
        self.prediction_param_group.layout().setContentsMargins(20, 20, 20, 10)

        # load 3D enabling checkbox
        self.enable_3d = QCheckBox('Enable 3D')
        self.enable_3d.setToolTip('Use a 3D model')
        self.prediction_param_group.layout().addWidget(self.enable_3d)

        # axes widget
        self.axes_widget = AxesWidget()
        self.prediction_param_group.layout().addWidget(self.axes_widget)

        self.layout().addWidget(self.prediction_param_group)

        ###############################
        # tiling
        self.tilling_group = QGroupBox()
        self.tilling_group.setTitle("Tiling (optional)")
        self.tilling_group.setLayout(QVBoxLayout())
        self.tilling_group.layout().setContentsMargins(20, 20, 20, 0)

        # checkbox
        self.tiling_cbox = QCheckBox('Tile prediction')
        self.tiling_cbox.setToolTip('Select to predict the image by tiles')
        self.tilling_group.layout().addWidget(self.tiling_cbox)

        # tiling spinbox
        self.tiling_spin = create_int_spinbox(1, 1000, 4, tooltip='Minimum number of tiles to use')
        self.tiling_spin.setEnabled(False)

        tiling_form = QFormLayout()
        tiling_form.addRow('Number of tiles', self.tiling_spin)
        tiling_widget = QWidget()
        tiling_widget.setLayout(tiling_form)
        self.tilling_group.layout().addWidget(tiling_widget)

        self.layout().addWidget(self.tilling_group)

        ###############################
        # threshold
        self.threshold_group = QGroupBox()
        self.threshold_group.setTitle("Threshold (optional)")
        self.threshold_group.setLayout(QVBoxLayout())
        self.threshold_group.layout().setContentsMargins(20, 20, 20, 0)

        # threshold slider
        self.threshold_cbox = QCheckBox('Apply threshold')
        self.threshold_cbox.setToolTip('Select to apply a threshold to the segmentation prediction')
        self.threshold_group.layout().addWidget(self.threshold_cbox)

        self.threshold_spin = threshold_spin()
        self.threshold_spin.native.setEnabled(False)
        self.threshold_spin.native.setToolTip('Threshold to use on all segmentation channels')
        self.threshold_group.layout().addWidget(self.threshold_spin.native)

        self.layout().addWidget(self.threshold_group)

        # prediction group
        self.prediction_group = QGroupBox()
        self.prediction_group.setTitle("Prediction")
        self.prediction_group.setLayout(QVBoxLayout())
        self.prediction_group.layout().setContentsMargins(20, 20, 20, 10)

        # progress bar
        self.pb_prediction = QProgressBar()
        self.pb_prediction.setMinimumHeight(30)
        self.pb_prediction.setValue(0)
        self.pb_prediction.setMinimum(0)
        self.pb_prediction.setMaximum(100)
        self.pb_prediction.setTextVisible(True)
        self.pb_prediction.setFormat(f'Images ?/?')
        self.prediction_group.layout().addWidget(self.pb_prediction)

        # predict button
        self.predict_button = QPushButton('Predict', self)
        self.predict_button.setToolTip('Start predicting')
        self.prediction_group.layout().addWidget(self.predict_button)
        self.layout().addWidget(self.prediction_group)

        # placeholders
        self.worker = None
        self.seg_prediction = None
        self.denoi_prediction = None

        # empty space
        # TODO place holder until we figure out how to not have the banner widget stretching
        empty_widget = QWidget()
        empty_widget.setMinimumHeight(80)
        self.layout().addWidget(empty_widget)

        # actions
        self.tabs.currentChanged.connect(self._update_tab_axes)
        self.predict_button.clicked.connect(self._start_prediction)
        self.images.changed.connect(self._update_layer_axes)
        self.images_folder.text_field.textChanged.connect(self._update_disk_axes)
        self.enable_3d.stateChanged.connect(self._update_3D)
        self.tiling_cbox.stateChanged.connect(self._update_tiling)
        self.threshold_cbox.stateChanged.connect(self._update_threshold)

        # members
        self.n_im = 0
        self.shape = None
        self.load_from_disk = 0

        # update axes widget in case of data
        self._update_layer_axes()

    def _update_threshold(self):
        self.threshold_spin.native.setEnabled(self.threshold_cbox.isChecked())

    def _update_tiling(self):
        self.tiling_spin.setEnabled(self.tiling_cbox.isChecked())

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
            # self.viewer.layers[SEGMENTATION].refresh()
            # self.viewer.layers[DENOISING].refresh()

        if UpdateType.DONE in updates:
            self.pb_prediction.setValue(100)
            self.pb_prediction.setFormat(f'Prediction done')

    def _start_prediction(self):
        if self.state == State.IDLE:
            if self.axes_widget.is_valid():
                if self.get_model_path().exists() and self.get_model_path().is_file():
                    # change state to running
                    self.state = State.RUNNING

                    # register which data tab: layers or disk
                    self.load_from_disk = self.tabs.currentIndex() == 1

                    self.predict_button.setText('Stop')

                    # remove seg and denoising layers if they are present
                    if SEGMENTATION in self.viewer.layers:
                        self.viewer.layers.remove(SEGMENTATION)
                    if DENOISING in self.viewer.layers:
                        self.viewer.layers.remove(DENOISING)

                    # reset place holders
                    self.seg_prediction = None
                    self.denoi_prediction = None

                    # start the prediction worker
                    self.worker = prediction_worker(self)
                    self.worker.yielded.connect(lambda x: self._update(x))
                    self.worker.returned.connect(self._done)
                    self.worker.start()
                else:
                    # ntf.show_error('Select a model')
                    ntf.show_info('Select a model')
            else:
                # ntf.show_error('Invalid axes')
                ntf.show_info('Invalid axes')

        elif self.state == State.RUNNING:
            # stop requested
            self.state = State.IDLE

    def _done(self):
        self.state = State.IDLE
        self.predict_button.setText('Predict again')

        if self.denoi_prediction is not None:
            self.viewer.add_image(self.denoi_prediction, name='denoised')

        if self.seg_prediction is not None:
            if self.threshold_cbox.isChecked():
                self.viewer.add_labels(self.seg_prediction, name='segmented', opacity=0.5)
            else:
                self.viewer.add_image(self.seg_prediction, name='segmented')

    def get_model_path(self):
        return self.load_model_button.Model.value

    def get_data_path(self):
        return self.images_folder.get_folder()

    # TODO call these methods throughout the workers
    def get_axes(self):
        return self.axes_widget.get_axes()

    def set_model_path(self, path: Path):
        self.load_model_button.Model.value = path

    def set_layer(self, layer):
        self.images.choices = [x for x in self.viewer.layers if type(x) is napari.layers.Image]
        if layer in self.images.choices:
            self.images.native.value = layer


class DemoPrediction(PredictWidgetWrapper):
    def __init__(self, napari_viewer):
        super().__init__(napari_viewer)

        # dowload demo files
        from napari_denoiseg._sample_data import demo_files
        ntf.show_info('Downloading data can take a few minutes.')

        # get files
        X, model = demo_files()

        # add image to viewer
        name = 'Demo images'
        napari_viewer.add_image(X, name=name)

        # modify path
        self.widget.set_model_path(model)
        self.widget.set_layer(name)


if __name__ == "__main__":
    from napari_denoiseg._sample_data import denoiseg_data_2D_n10

    data = denoiseg_data_2D_n10()

    # create a Viewer
    viewer = napari.Viewer()

    # add our plugin
    viewer.window.add_dock_widget(PredictWidgetWrapper(viewer))

    # add images
    viewer.add_image(data[0][0][0:30], name=data[0][1]['name'])

    napari.run()
