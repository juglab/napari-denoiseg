from pathlib import Path
from qtpy.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QPushButton,
    QFormLayout,
    QTableWidget,
    QTabWidget,
    QTableWidgetItem,
    QHeaderView,
    QCheckBox
)
import napari
from napari_denoiseg.widgets import FolderWidget, AxesWidget, two_layers_choice, load_button
from napari_denoiseg.utils import State, optimizer_worker, loading_worker

T = 't'
M = 'metrics'
SAMPLE = 'Sample data'


class ThresholdWidget(QWidget):
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
        self.layer_choice = two_layers_choice()
        self.images = self.layer_choice.Images
        self.labels = self.layer_choice.Labels
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
        self.images.choices = [x for x in napari_viewer.layers if type(x) is napari.layers.Image]
        self.labels.choices = [x for x in napari_viewer.layers if type(x) is napari.layers.Labels]

        ###############################
        # other

        # load model button
        self.load_model_button = load_button()
        self.layout().addWidget(self.load_model_button.native)

        # load 3D enabling checkbox
        self.enable_3d = QCheckBox('Enable 3D')
        self.layout().addWidget(self.enable_3d)

        # axes widget
        self.axes_widget = AxesWidget()
        self.layout().addWidget(self.axes_widget)

        # feedback table to users
        self.table = QTableWidget()
        self.table.setRowCount(19)
        self.table.setColumnCount(2)

        self.table.setHorizontalHeaderLabels([T, M])
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.table.resizeRowsToContents()
        self.layout().addWidget(self.table)

        # optimize button
        self.optimize_button = QPushButton("Optimize", self)
        self.layout().addWidget(self.optimize_button)

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
            if self.axes_widget.is_valid() and Path(self.get_model_path()).exists():
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
                # TODO: feedback to user?
                pass
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
    from napari_denoiseg._sample_data import denoiseg_data_2D_n10

    data = denoiseg_data_2D_n10()

    # create a Viewer
    viewer = napari.Viewer()

    # add our plugin
    viewer.window.add_dock_widget(ThresholdWidget(viewer))

    # add images
    viewer.add_image(data[0][0][0:15], name=data[0][1]['name'])
    viewer.add_labels(data[1][0][0:15], name=data[1][1]['name'])

    napari.run()
