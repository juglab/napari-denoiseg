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
import napari
from napari_denoiseg._train_widget import State
from napari_denoiseg.utils import FolderWidget, two_layers_choice, load_button
from napari_denoiseg.utils import optimizer_worker

T = 't'
M = 'metrics'


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
        self.images.choices = [x for x in napari_viewer.layers if type(x) is napari.layers.Image]
        self.labels.choices = [x for x in napari_viewer.layers if type(x) is napari.layers.Labels]

        ###############################
        # other

        # load model button
        self.load_button = load_button()
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

        # optimize button
        self.optimize_button = QPushButton("Optimize", self)
        self.optimize_button.clicked.connect(self.start_optimization)
        self.layout().addWidget(self.optimize_button)

        # set variables
        self.worker = None
        self.load_from_disk = 0

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


if __name__ == "__main__":
    from napari_denoiseg._sample_data import denoiseg_data_n0

    data = denoiseg_data_n0()

    # create a Viewer
    viewer = napari.Viewer()

    # add our plugin
    viewer.window.add_dock_widget(ThresholdWidget(viewer))

    # add images
    viewer.add_image(data[0][0][0:15], name=data[0][1]['name'])
    viewer.add_labels(data[1][0][0:15], name=data[1][1]['name'])

    napari.run()
