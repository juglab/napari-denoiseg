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
from napari_denoiseg.widgets import enable_3d
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
        self.labels = self.layer_choice.Labels
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
        self.patch_size_XY = QSpinBox()
        self.patch_size_XY.setMaximum(512)
        self.patch_size_XY.setMinimum(16)
        self.patch_size_XY.setSingleStep(8)
        self.patch_size_XY.setValue(16)

        # 3D checkbox
        self.enable_3d = enable_3d()
        self.patch_size_Z = QSpinBox()
        self.patch_size_Z.setMaximum(512)
        self.patch_size_Z.setMinimum(16)
        self.patch_size_Z.setSingleStep(8)
        self.patch_size_Z.setValue(16)
        self.patch_size_Z.setVisible(False)
        self.patch_size_Z_label = QLabel()
        self.patch_size_Z_label.setText("Patch Z")
        self.patch_size_Z_label.setVisible(False)

        # TODO add tooltips
        others = QWidget()
        formLayout = QFormLayout()
        formLayout.addRow('', self.axes_widget)
        formLayout.addRow('Enable 3D', self.enable_3d.native)
        formLayout.addRow('N epochs', self.n_epochs_spin)
        formLayout.addRow('N steps', self.n_steps_spin)
        formLayout.addRow('Batch size', self.batch_size_spin)
        formLayout.addRow('Patch XY', self.patch_size_XY)
        formLayout.addRow(self.patch_size_Z_label, self.patch_size_Z)
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
        self.zero_model_button = QPushButton("Zero model", self)
        self.zero_model_button.setEnabled(False)
        self.zero_model_button.setVisible(False)

        train_buttons.layout().addWidget(self.zero_model_button)
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
        self.plot = TBPlotWidget(max_width=300, max_height=300, min_height=250)
        self.layout().addWidget(self.plot.native)

        # place-holder for models and parameters (e.g. bioimage.io)
        self.worker = None
        self.model, self.threshold = None, None
        self.inputs, self.outputs = [], []
        self.tf_version = None
        self.load_from_disk = False

        # actions
        self.tabs.currentChanged.connect(self._update_tab_axes)
        self.enable_3d.use3d.changed.connect(self._update_3D)
        self.images.changed.connect(self._update_layer_axes)
        self.train_images_folder.text_field.textChanged.connect(self._update_disk_axes)
        self.train_button.clicked.connect(lambda: self._start_training(self.model))
        self.zero_model_button.clicked.connect(self._zero_model)
        self.n_epochs_spin.valueChanged.connect(self._update_epochs)
        self.n_steps_spin.valueChanged.connect(self._update_steps)
        self.save_button.clicked.connect(self._save_model)

        # update axes widget in case of data
        self._update_layer_axes()

    def _start_training(self, pretrained_model=None):
        """
        Start training from scratch or using a pretrained model.

        :param pretrained_model:
        :return:
        """
        if self.state == State.IDLE:
            # TODO check that all in order before predicting (data loaded, axes valid ...etc...)

            if self.axes_widget.is_valid():
                self.state = State.RUNNING

                # register which data tab: layers or disk
                self.load_from_disk = self.tabs.currentIndex() == 1

                # modify UI
                self.plot.clear_plot()
                self.threshold_label.setText("Best threshold: ?")
                self.train_button.setText('Stop')
                self.zero_model_button.setEnabled(False)
                self.zero_model_button.setVisible(False)
                self.save_button.setEnabled(False)

                # instantiate worker and start training
                self.worker = training_worker(self, pretrained_model=pretrained_model)
                self.worker.yielded.connect(lambda x: self._update_all(x))
                self.worker.returned.connect(self._done)
                self.worker.start()
            else:
                # TODO feedback to users?
                pass

    def _done(self):
        """
        Called when training is finished.

        :return:
        """
        self.state = State.IDLE
        self.train_button.setText('Train new')
        self.zero_model_button.setEnabled(True)
        self.zero_model_button.setVisible(True)

        self.threshold_label.setText("Best threshold: {:.2f}".format(self.threshold))

        self.save_button.setEnabled(True)

    def _zero_model(self):
        """
        Zero the model, causing the next training session to train from scratch.
        :return:
        """
        if self.state == State.IDLE:
            self.model = None

    def _update_3D(self, event):
        """
        Update the UI based on the status of the 3D checkbox.
        :param event:
        :return:
        """
        self.is_3D = event.value
        self.patch_size_Z.setVisible(self.is_3D)
        self.patch_size_Z_label.setVisible(self.is_3D)

        # update axes widget
        self.axes_widget.update_is_3D(self.is_3D)
        self.axes_widget.set_text_field(self.axes_widget.get_default_text())

    def _update_layer_axes(self):
        """
        Update the axes widget based on the shape of the data selected in the layer selection drop-down widget.
        :return:
        """
        if self.images.value is not None:
            shape = self.images.value.data.shape

            # update shape length in the axes widget
            self.axes_widget.update_axes_number(len(shape))
            self.axes_widget.set_text_field(self.axes_widget.get_default_text())

    def _update_disk_axes(self):
        """
        Load an example image from the disk and update the axes widget based on its shape.

        :return:
        """
        def add_image(widget, image):
            if image is not None:
                if SAMPLE in widget.viewer.layers:
                    widget.viewer.layers.remove(SAMPLE)

                widget.viewer.add_image(image, name=SAMPLE, visible=True)

                # update the axes widget
                widget.axes_widget.update_axes_number(len(image.shape))
                widget.axes_widget.set_text_field(widget.axes_widget.get_default_text())

        path = self.train_images_folder.get_folder()

        if path is not None or path != '':
            # load one image
            load_worker = loading_worker(path)
            load_worker.yielded.connect(lambda x: add_image(self, x))
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

    def _update_epochs(self):
        """
        Update the epoch progress bar following a change of total number of epochs.
        :return:
        """
        if self.state == State.IDLE:
            self.n_epochs = self.n_epochs_spin.value()
            self.pb_epochs.setValue(0)
            self.pb_epochs.setFormat(f'Epoch ?/{self.n_epochs_spin.value()}')

    def _update_steps(self):
        """
        Update the step progress bar following a change of total number of steps.
        :return:
        """
        if self.state == State.IDLE:
            self.n_steps = self.n_steps_spin.value()
            self.pb_steps.setValue(0)
            self.pb_steps.setFormat(f'Step ?/{self.n_steps_spin.value()}')

    def _update_all(self, updates):
        """
        Update the UI following an update event from the training worker.
        :param updates:
        :return:
        """
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
        """
        Export the model.
        :return:
        """
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
    from napari_denoiseg._sample_data import denoiseg_data_2D_n0, denoiseg_data_3D

    dims = '3D'  # '2D'
    if dims == '3D':
        data = denoiseg_data_3D()
    else:
        data = denoiseg_data_2D_n0()

    # create a Viewer
    viewer = napari.Viewer()

    # add images
    viewer.add_image(data[0][0][0:60], name=data[0][1]['name'])
    viewer.add_labels(data[1][0][0:15], name=data[1][1]['name'])

    # add our plugin
    viewer.window.add_dock_widget(TrainWidget(viewer))

    napari.run()
