from qtpy.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QPushButton,
    QFormLayout,
    QTableWidget,
    QTabWidget,
    QTableWidgetItem,
    QHeaderView,
    QCheckBox,
    QGroupBox
)
from qtpy.QtCore import Qt

import napari
import napari.utils.notifications as ntf

from napari_denoiseg.widgets import (
    FolderWidget,
    BannerWidget,
    ScrollWidgetWrapper,
    AxesWidget,
    create_gpu_label,
    two_layers_choice,
    load_button
)
from napari_denoiseg.utils import State, optimizer_worker, loading_worker
from napari_denoiseg.resources import ICON_JUGLAB

T = 't'
M = 'metrics'
SAMPLE = 'Sample data'


class ThresholdWidgetWrapper(ScrollWidgetWrapper):
    def __init__(self, napari_viewer):
        super().__init__(ThresholdWidget(napari_viewer))


class ThresholdWidget(QWidget):
    def __init__(self, napari_viewer: napari.Viewer):
        super().__init__()

        self.state = State.IDLE
        self.viewer = napari_viewer

        self.setLayout(QVBoxLayout())

        ###############################
        # Banner
        self.layout().addWidget(BannerWidget('DenoiSeg - Threshold',
                                             ICON_JUGLAB,
                                             'A joint denoising and segmentation algorithm requiring '
                                             'only a few annotated ground truth images.',
                                             'https://juglab.github.io/napari_denoiseg',
                                             'https://github.com/juglab/napari_denoiseg/issues'))

        # add GPU button
        gpu_button = create_gpu_label()
        gpu_button.setAlignment(Qt.AlignmentFlag.AlignRight)
        self.layout().addWidget(gpu_button)

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

        self.tabs.setTabToolTip(0, 'Use images from napari layers')
        self.tabs.setTabToolTip(1, 'Use images saved on the disk')

        # layer tabs
        self.layer_choice = two_layers_choice()
        self.images = self.layer_choice.Images
        self.labels = self.layer_choice.Labels
        tab_layers.layout().addWidget(self.layer_choice.native)

        self.images.native.setToolTip('Select an image layer')
        self.labels.native.setToolTip('Select a label layer corresponding to the image one')

        # disk tab
        self.images_folder = FolderWidget('Choose')
        self.labels_folder = FolderWidget('Choose')

        self.images_folder.setToolTip('Select a folder containing the images')
        self.labels_folder.setToolTip('Select a folder containing the ground-truths')

        buttons = QWidget()
        form = QFormLayout()

        form.addRow('Images', self.images_folder)
        form.addRow('Labels', self.labels_folder)

        buttons.setLayout(form)
        tab_disk.layout().addWidget(buttons)

        # add to main layout
        self.layout().addWidget(self.tabs)
        self.images.choices = [x for x in napari_viewer.layers if type(x) is napari.layers.Image]
        self.labels.choices = [x for x in napari_viewer.layers if type(x) is napari.layers.Labels]

        ###############################
        # other

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
        # Parameters
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
        self.layout().addWidget(self.axes_widget)
        self.prediction_param_group.layout().addWidget(self.axes_widget)

        self.layout().addWidget(self.prediction_param_group)

        # optimize button
        self.optimize_button = QPushButton("Optimize", self)
        self.optimize_button.setToolTip('Find the optimum threshold')
        self.layout().addWidget(self.optimize_button)

        # feedback table to users
        self.results_group = QGroupBox()
        self.results_group.setTitle("Results")
        self.results_group.setLayout(QVBoxLayout())
        self.results_group.layout().setContentsMargins(20, 20, 20, 10)

        self.table = QTableWidget()
        self.table.setRowCount(19)
        self.table.setColumnCount(2)
        self.table.setMinimumHeight(400)

        self.table.setHorizontalHeaderLabels([T, M])
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.table.resizeRowsToContents()

        self.results_group.layout().addWidget(self.table)
        self.layout().addWidget(self.results_group)

        # set variables
        self.worker = None
        self.load_from_disk = 0

        # actions
        self.tabs.currentChanged.connect(self._update_tab_axes)
        self.optimize_button.clicked.connect(self._start_optimization)
        self.images_folder.text_field.textChanged.connect(self._update_disk_axes)
        self.images.changed.connect(self._update_layer_axes)
        self.enable_3d.stateChanged.connect(self._update_3D)

        # update axes widget in case of data
        self._update_layer_axes()

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

    def _start_optimization(self):
        if self.state == State.IDLE:
            if self.axes_widget.is_valid():
                if self.get_model_path().exists() and self.get_model_path().is_file():
                    self.state = State.RUNNING

                    # register which data tab: layers or disk
                    self.load_from_disk = self.tabs.currentIndex() == 1

                    self.optimize_button.setText('Stop')
                    self.table.clearContents()

                    self.worker = optimizer_worker(self)
                    self.worker.returned.connect(self._done)
                    self.worker.yielded.connect(lambda x: self._update(x))
                    self.worker.start()
                else:
                    # ntf.show_error('Select a model')
                    ntf.show_info('Select a model')
            else:
                # ntf.show_error('Invalid axes')
                ntf.show_info('Invalid axes')
        elif self.state == State.RUNNING:
            self.state = State.IDLE

    def _update(self, score_tuple):
        (i, t, m) = score_tuple
        self.table.setItem(i, 0, QTableWidgetItem("{:.2f}".format(t)))
        self.table.setItem(i, 1, QTableWidgetItem("{:.2f}".format(m)))

    def _done(self):
        self.state = State.IDLE
        self.optimize_button.setText('Optimize')

    def get_model_path(self):
        return self.load_model_button.Model.value


if __name__ == "__main__":
    from napari_denoiseg._sample_data import denoiseg_data_2D_n20

    data = denoiseg_data_2D_n20()

    # create a Viewer
    viewer = napari.Viewer()

    # add our plugin
    viewer.window.add_dock_widget(ThresholdWidgetWrapper(viewer))

    # add images
    viewer.add_image(data[0][0][0:15], name=data[0][1]['name'])
    viewer.add_labels(data[1][0][0:15], name=data[1][1]['name'])

    napari.run()
