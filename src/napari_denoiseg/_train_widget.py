"""
"""
import napari
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
from napari_denoiseg.widgets import TBPlotWidget, FolderWidget, AxesWidget
from napari_denoiseg.widgets import two_layers_choice, percentage_slider
from napari_denoiseg.utils import State, UpdateType, ModelSaveMode
from napari_denoiseg.utils import training_worker, loading_worker


SAMPLE = 'Sample data'


class TrainWidget(QWidget):
    def __init__(self, napari_viewer):
        super().__init__()

        self.state = State.IDLE
        self.viewer = napari_viewer

        self.setLayout(QVBoxLayout())

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
        self.tabs.setMaximumHeight(200)

        # layer tabs
        self.layer_choice = two_layers_choice()
        self.images = self.layer_choice.Images
        self.labels = self.layer_choice.Masks
        tab_layers.layout().addWidget(self.layer_choice.native)

        self.perc_train_slider = percentage_slider()
        perc_widget = QWidget()
        perc_widget.setLayout(QFormLayout())
        perc_widget.layout().addRow('Train label %', self.perc_train_slider.native)
        tab_layers.layout().addWidget(perc_widget)

        # disk tab
        self.train_images_folder = FolderWidget('Choose')
        self.train_labels_folder = FolderWidget('Choose')
        self.val_images_folder = FolderWidget('Choose')
        self.val_labels_folder = FolderWidget('Choose')

        buttons = QWidget()
        form = QFormLayout()

        form.addRow('Train images', self.train_images_folder)
        form.addRow('Train labels', self.train_labels_folder)
        form.addRow('Val images', self.val_images_folder)
        form.addRow('Val labels', self.val_labels_folder)

        buttons.setLayout(form)
        tab_disk.layout().addWidget(buttons)

        # add to main layout
        self.layout().addWidget(self.tabs)
        self.images.choices = [x for x in napari_viewer.layers if type(x) is napari.layers.Image]
        self.labels.choices = [x for x in napari_viewer.layers if type(x) is napari.layers.Labels]

        ###############################
        # axes
        self.axes_widget = AxesWidget()
        #self.layout().addWidget(self.axes_widget)

        # others
        self.n_epochs_spin = QSpinBox()
        self.n_epochs_spin.setMinimum(1)
        self.n_epochs_spin.setMaximum(1000)
        self.n_epochs_spin.setValue(2)
        self.n_epochs = self.n_epochs_spin.value()

        self.n_steps_spin = QSpinBox()
        self.n_steps_spin.setMaximum(1000)
        self.n_steps_spin.setMinimum(1)
        self.n_steps_spin.setValue(10)
        self.n_steps = self.n_steps_spin.value()

        # batch size
        self.batch_size_spin = QSpinBox()
        self.batch_size_spin.setMaximum(512)
        self.batch_size_spin.setMinimum(0)
        self.batch_size_spin.setSingleStep(8)
        self.batch_size_spin.setValue(16)

        # patch size
        self.patch_size_spin = QSpinBox()
        self.patch_size_spin.setMaximum(512)
        self.patch_size_spin.setMinimum(16)
        self.patch_size_spin.setSingleStep(8)
        self.patch_size_spin.setValue(16)

        # TODO add tooltips
        others = QWidget()
        formLayout = QFormLayout()
        formLayout.addRow('', self.axes_widget)
        formLayout.addRow('N epochs', self.n_epochs_spin)
        formLayout.addRow('N steps', self.n_steps_spin)
        formLayout.addRow('Batch size', self.batch_size_spin)
        formLayout.addRow('Patch XY', self.patch_size_spin)
        others.setLayout(formLayout)
        self.layout().addWidget(others)

        # progress bars
        progress_widget = QWidget()
        progress_widget.setLayout(QVBoxLayout())

        self.pb_epochs = QProgressBar()
        self.pb_epochs.setValue(0)
        self.pb_epochs.setMinimum(0)
        self.pb_epochs.setMaximum(100)
        self.pb_epochs.setTextVisible(True)
        self.pb_epochs.setFormat(f'Epoch ?/{self.n_epochs_spin.value()}')

        self.pb_steps = QProgressBar()
        self.pb_steps.setValue(0)
        self.pb_steps.setMinimum(0)
        self.pb_steps.setMaximum(100)
        self.pb_steps.setTextVisible(True)
        self.pb_steps.setFormat(f'Step ?/{self.n_steps_spin.value()}')

        progress_widget.layout().addWidget(self.pb_epochs)
        progress_widget.layout().addWidget(self.pb_steps)
        self.layout().addWidget(progress_widget)

        # train button
        train_buttons = QWidget()
        train_buttons.setLayout(QHBoxLayout())

        self.train_button = QPushButton("Train", self)
        self.retrain_button = QPushButton("", self)
        self.retrain_button.setEnabled(False)

        train_buttons.layout().addWidget(self.retrain_button)
        train_buttons.layout().addWidget(self.train_button)

        self.layout().addWidget(train_buttons)

        # Threshold
        self.threshold_label = QLabel()
        self.threshold_label.setText("Best threshold: ?")
        self.layout().addWidget(self.threshold_label)

        # Save button
        save_widget = QWidget()
        save_widget.setLayout(QHBoxLayout())

        self.save_choice = QComboBox()
        self.save_choice.addItems(ModelSaveMode.list())

        self.save_button = QPushButton("Save model", self)
        self.save_button.setEnabled(False)

        save_widget.layout().addWidget(self.save_button)
        save_widget.layout().addWidget(self.save_choice)
        self.layout().addWidget(save_widget)

        # plot widget
        self.plot = TBPlotWidget(max_width=300, max_height=300)
        self.layout().addWidget(self.plot.native)

        # worker
        self.worker = None

        # actions
        self.images.changed.connect(self._update_layer_axes)
        self.train_images_folder.text_field.textChanged.connect(self._update_disk_axes)
        self.train_button.clicked.connect(self._start_training)
        self.retrain_button.clicked.connect(self._continue_training)
        self.n_epochs_spin.valueChanged.connect(self._update_epochs)
        self.n_steps_spin.valueChanged.connect(self._update_steps)
        self.save_button.clicked.connect(self._save_model)

        # place-holder for models and parameters (e.g. bioimage.io)
        self.model, self.threshold = None, None
        self.inputs, self.outputs = [], []
        self.tf_version = None
        self.load_from_disk = False

        # update axes widget in case of data
        self._update_layer_axes()

    def _start_training(self, pretrained_model=None):
        if self.state == State.IDLE:
            if self.axes_widget.is_valid():
                self.state = State.RUNNING

                # register which data tab: layers or disk
                self.load_from_disk = self.tabs.currentIndex() == 1

                # modify UI
                self.plot.clear_plot()
                self.threshold_label.setText("Best threshold: ?")
                self.train_button.setText('Stop')
                self.retrain_button.setText('')
                self.retrain_button.setEnabled(False)
                self.save_button.setEnabled(False)

                # instantiate worker and start training
                self.worker = training_worker(self, pretrained_model=pretrained_model)
                self.worker.yielded.connect(lambda x: self._update_all(x))
                self.worker.returned.connect(self._done)
                self.worker.start()
            else:
                # TODO feedback to users?
                pass
        elif self.state == State.RUNNING:
            self.state = State.IDLE

    def _continue_training(self):
        if self.state == State.IDLE:
            self.start_training(pretrained_model=self.model)

    def _done(self):
        self.state = State.IDLE
        self.train_button.setText('Train new')
        self.retrain_button.setText('Retrain')
        self.retrain_button.setEnabled(True)

        self.threshold_label.setText("Best threshold: {:.2f}".format(self.threshold))

        self.save_button.setEnabled(True)

    def _update_3D(self):
        # TODO: get checkbox state and pass it on to the axes
        pass

    def _update_layer_axes(self):
        if self.images.value.data is not None:
            shape = self.images.value.data.shape

            # update shape length in the axes widget
            self.axes_widget.update_axes_number(len(shape))
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
        path = self.train_images_folder.get_folder()

        # load one image
        load_worker = loading_worker(path)
        load_worker.yielded.connect(lambda x: self._add_image(x))
        load_worker.start()

    def _update_epochs(self):
        if self.state == State.IDLE:
            self.n_epochs = self.n_epochs_spin.value()
            self.pb_epochs.setValue(0)
            self.pb_epochs.setFormat(f'Epoch ?/{self.n_epochs_spin.value()}')

    def _update_steps(self):
        if self.state == State.IDLE:
            self.n_steps = self.n_steps_spin.value()
            self.pb_steps.setValue(0)
            self.pb_steps.setFormat(f'Step ?/{self.n_steps_spin.value()}')

    def _update_all(self, updates):
        if self.state == State.RUNNING:
            if UpdateType.EPOCH in updates:
                val = updates[UpdateType.EPOCH]
                e_perc = int(100 * updates[UpdateType.EPOCH] / self.n_epochs + 0.5)
                self.pb_epochs.setValue(e_perc)
                self.pb_epochs.setFormat(f'Epoch {val}/{self.n_epochs}')

            if UpdateType.BATCH in updates:
                val = updates[UpdateType.BATCH]
                s_perc = int(100 * val / self.n_steps + 0.5)
                self.pb_steps.setValue(s_perc)
                self.pb_steps.setFormat(f'Step {val}/{self.n_steps}')

            if UpdateType.LOSS in updates:
                self.plot.update_plot(*updates[UpdateType.LOSS])

    def _save_model(self):
        if self.state == State.IDLE:
            if self.model:
                where = QFileDialog.getSaveFileName(caption='Save model')[0]

                export_type = self.save_choice.currentText()
                if ModelSaveMode.MODELZOO.value == export_type:
                    from napari_denoiseg.utils import build_modelzoo
                    build_modelzoo(where + '.bioimage.io.zip',
                                   self.model.logdir / "weights_best.h5",
                                   self.inputs,
                                   self.outputs,
                                   self.tf_version)
                else:
                    self.model.keras_model.save_weights(where + '.h5')


if __name__ == "__main__":
    from napari_denoiseg._sample_data import denoiseg_data_n0

    data = denoiseg_data_n0()

    # create a Viewer
    viewer = napari.Viewer()

    # add images
    viewer.add_image(data[0][0][0:60], name=data[0][1]['name'])
    viewer.add_labels(data[1][0][0:15], name=data[1][1]['name'])

    # add our plugin
    viewer.window.add_dock_widget(TrainWidget(viewer))

    napari.run()
