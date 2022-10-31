"""
"""
from pathlib import Path

import napari
import napari.utils.notifications as ntf

from qtpy import QtGui
from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QPushButton,
    QFormLayout,
    QComboBox,
    QFileDialog,
    QTabWidget,
    QGroupBox
)
from napari_denoiseg.utils import (
    State,
    UpdateType,
    ModelSaveMode,
    training_worker,
    loading_worker,
    save_model
)
from napari_denoiseg.widgets import (
    TBPlotWidget,
    FolderWidget,
    AxesWidget,
    BannerWidget,
    TrainingSettingsWidget,
    ScrollWidgetWrapper,
    enable_3d,
    create_int_spinbox,
    create_progressbar,
    create_gpu_label,
    two_layers_choice,
    percentage_slider
)
from napari_denoiseg.resources import ICON_GEAR, ICON_JUGLAB

SAMPLE = 'Example data'


class TrainingWidgetWrapper(ScrollWidgetWrapper):
    def __init__(self, napari_viewer):
        super().__init__(TrainWidget(napari_viewer))


class TrainWidget(QWidget):
    def __init__(self, napari_viewer: napari.Viewer):
        super().__init__()
        self.state = State.IDLE
        self.viewer = napari_viewer
        self.setLayout(QVBoxLayout())
        self.setMinimumWidth(200)

        # add banner
        self.layout().addWidget(BannerWidget('DenoiSeg - Training',
                                             ICON_JUGLAB,
                                             'A joint denoising and segmentation algorithm requiring '
                                             'only a few annotated ground truth images.',
                                             'https://juglab.github.io/napari_denoiseg',
                                             'https://github.com/juglab/napari_denoiseg/issues'))

        # add GPU button
        gpu_button = create_gpu_label()
        gpu_button.setAlignment(Qt.AlignmentFlag.AlignRight)
        self.layout().addWidget(gpu_button)

        # other widgets
        self._build_data_selection_widgets(napari_viewer)
        self._build_training_param_widgets()
        self._build_train_widgets()
        self._build_save_widgets()
        self._build_progress_widgets()
        self.expert_settings = None

        # place-holder for models and parameters (e.g. bioimage.io)
        self.is_3D = False
        self.worker = None
        self.model = None
        self.inputs, self.outputs = [], []
        self.tf_version = None
        self.load_from_disk = False
        self.training_done = False
        self.training_arguments = None
        self.X_val = None
        self.reset = True

        # actions
        self._set_actions()

        # update axes widget in case of data
        self._update_layer_axes()

    def _set_actions(self):
        # actions
        self.tabs.currentChanged.connect(self._update_tab_axes)
        self.enable_3d.use3d.changed.connect(self._update_3D)
        self.images.changed.connect(self._update_layer_axes)
        self.train_images_folder.text_field.textChanged.connect(self._update_disk_axes)
        self.train_button.clicked.connect(lambda: self._start_training(self.model))
        self.reset_model_button.clicked.connect(self._reset_model)
        self.n_epochs_spin.valueChanged.connect(self._update_epochs)
        self.n_steps_spin.valueChanged.connect(self._update_steps)
        self.save_button.clicked.connect(self._save_model)

    def _build_train_widgets(self):
        self.train_group = QGroupBox()
        self.train_group.setTitle("Train")
        self.train_group.setLayout(QVBoxLayout())

        # train button
        train_buttons = QWidget()
        train_buttons.setLayout(QHBoxLayout())

        self.train_button = QPushButton('Train', self)

        self.reset_model_button = QPushButton('', self)
        self.reset_model_button.setEnabled(False)
        self.reset_model_button.setToolTip('Reset the weights of the model (forget the training)')

        train_buttons.layout().addWidget(self.reset_model_button)
        train_buttons.layout().addWidget(self.train_button)
        self.train_group.layout().addWidget(train_buttons)

        self.layout().addWidget(self.train_group)

    def _build_progress_widgets(self):
        self.progress_group = QGroupBox()
        self.progress_group.setTitle("Training progress")

        # progress bars
        self.progress_group.setLayout(QVBoxLayout())
        self.progress_group.layout().setContentsMargins(20, 20, 20, 0)

        self.pb_epochs = create_progressbar(max_value=self.get_n_epochs(),
                                            text_format=f'Epoch ?/{self.get_n_epochs()}')

        self.pb_steps = create_progressbar(max_value=self.n_steps_spin.value(),
                                           text_format=f'Step ?/{self.n_steps_spin.value()}')

        self.progress_group.layout().addWidget(self.pb_epochs)
        self.progress_group.layout().addWidget(self.pb_steps)

        # plot widget
        self.plot = TBPlotWidget(max_width=300, max_height=300, min_height=250)
        self.progress_group.layout().addWidget(self.plot.native)
        self.layout().addWidget(self.progress_group)

    def _build_data_selection_widgets(self, napari_viewer):
        # QTabs
        self.tabs = QTabWidget()
        tab_layers = QWidget()
        tab_layers.setLayout(QVBoxLayout())
        tab_disk = QWidget()
        tab_disk.setLayout(QVBoxLayout())

        # add tabs
        self.tabs.addTab(tab_layers, 'From layers')
        self.tabs.addTab(tab_disk, 'From disk')
        self.tabs.setTabToolTip(0, 'Use images from napari layers')
        self.tabs.setTabToolTip(1, 'Use patches saved on the disk')
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

        self.images.native.setToolTip('Select an image layer')
        self.labels.native.setToolTip('Select a label layer corresponding to the images')

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

        self.train_images_folder.setToolTip('Select a folder containing the training images')
        self.train_labels_folder.setToolTip('Select a folder containing the training ground-truth')
        self.val_images_folder.setToolTip('Select a folder containing the validation images')
        self.val_labels_folder.setToolTip('Select a folder containing the validation ground-truth')

        # add to main layout
        self.layout().addWidget(self.tabs)
        self.images.choices = [x for x in napari_viewer.layers if type(x) is napari.layers.Image]
        self.labels.choices = [x for x in napari_viewer.layers if type(x) is napari.layers.Labels]

    def _build_training_param_widgets(self):
        self.training_param_group = QGroupBox()
        self.training_param_group.setTitle("Training parameters")
        self.training_param_group.setMinimumWidth(100)

        # expert settings
        icon = QtGui.QIcon(ICON_GEAR)
        self.training_expert_btn = QPushButton(icon, '')
        self.training_expert_btn.clicked.connect(self._training_expert_setter)
        self.training_expert_btn.setFixedSize(30, 30)
        self.training_expert_btn.setToolTip('Open the expert settings menu')

        # axes
        self.axes_widget = AxesWidget()

        # others
        self.n_epochs_spin = create_int_spinbox(1, 1000, 30, tooltip='Number of epochs')
        self.n_epochs = self.get_n_epochs()

        self.n_steps_spin = create_int_spinbox(1, 1000, 200, tooltip='Number of steps per epochs')
        self.n_steps = self.n_steps_spin.value()

        # batch size
        self.batch_size_spin = create_int_spinbox(1, 512, 16, 1, tooltip='Number of patches per batch (decrease if '
                                                                         'GPU memory is insufficient)')

        # patch size
        self.patch_size_XY = create_int_spinbox(16, 512, 64, 8, tooltip='Dimension of the patches in XY')

        # 3D checkbox
        self.enable_3d = enable_3d()
        self.enable_3d.native.setToolTip('Use a 3D network')
        self.patch_size_Z = create_int_spinbox(16, 512, 32, 8, False, tooltip='Dimension of the patches in Z')

        formLayout = QFormLayout()
        formLayout.addRow(self.axes_widget.label.text(), self.axes_widget.text_field)
        formLayout.addRow('Enable 3D', self.enable_3d.native)
        formLayout.addRow('N epochs', self.n_epochs_spin)
        formLayout.addRow('N steps', self.n_steps_spin)
        formLayout.addRow('Batch size', self.batch_size_spin)
        formLayout.addRow('Patch XY', self.patch_size_XY)
        formLayout.addRow('Patch Z', self.patch_size_Z)
        formLayout.minimumSize()
        hlayout = QVBoxLayout()
        hlayout.addWidget(self.training_expert_btn, alignment=Qt.AlignRight | Qt.AlignVCenter)
        hlayout.addLayout(formLayout)

        self.training_param_group.setLayout(hlayout)
        self.training_param_group.layout().setContentsMargins(5, 20, 5, 10)
        self.layout().addWidget(self.training_param_group)

    def _build_save_widgets(self):
        self.save_group = QGroupBox()
        self.save_group.setTitle("Save")
        self.save_group.setLayout(QVBoxLayout())

        # Save button
        save_widget = QWidget()
        save_widget.setLayout(QHBoxLayout())
        self.save_choice = QComboBox()
        self.save_choice.addItems(ModelSaveMode.list())
        self.save_choice.setToolTip('Output format')

        self.save_button = QPushButton("Save model", self)
        self.save_button.setEnabled(False)
        self.save_choice.setToolTip('Save the model weights and configuration')

        save_widget.layout().addWidget(self.save_button)
        save_widget.layout().addWidget(self.save_choice)
        self.save_group.layout().addWidget(save_widget)

        self.layout().addWidget(self.save_group)

    def _start_training(self, pretrained_model=None):
        """
        Start training from scratch or using a pretrained model.

        :param pretrained_model:
        :return:
        """
        if self.state == State.IDLE:
            if self.axes_widget.is_valid():

                # set the state to running
                self.state = State.RUNNING
                self.training_done = False

                # register which data tab: layers or disk
                self.load_from_disk = self.tabs.currentIndex() == 1

                # modify UI
                if self.reset:
                    # start from scratch
                    self.plot.clear_plot()

                    # reset epochs and steps
                    self._update_steps(self.n_steps_spin.value())
                    self._update_epochs(self.get_n_epochs())

                self.reset = False
                self.train_button.setText('Stop')
                self.reset_model_button.setText('')
                self.reset_model_button.setEnabled(False)
                self.save_button.setEnabled(False)

                # instantiate worker and start training
                self.worker = training_worker(self,
                                              pretrained_model=pretrained_model,
                                              expert_settings=self.expert_settings)
                self.worker.yielded.connect(lambda x: self._update_all(x))
                self.worker.returned.connect(self._done)
                self.worker.start()
            else:
                # ntf.show_error('Invalid axes')
                ntf.show_info('Invalid axes')
        else:
            self.state = State.IDLE

    def _done(self):
        """
        Called when training is finished.

        :return:
        """
        self.state = State.IDLE
        self.reset = False
        self.train_button.setText('Continue training')
        self.reset_model_button.setText('Reset model')
        self.reset_model_button.setEnabled(True)
        self.save_button.setEnabled(True)

    def _reset_model(self):
        """
        Reset the model, causing the next training session to train from scratch.
        :return:
        """
        if self.state == State.IDLE:
            self.reset = True
            self.model = None
            self.reset_model_button.setText('')
            self.reset_model_button.setEnabled(False)
            self.save_button.setEnabled(False)
            self.train_button.setText('Train')

    def _update_3D(self, val):
        """
        Update the UI based on the status of the 3D checkbox.
        :param val:
        :return:
        """
        # TODO in magicgui 0.4 even becomes a Bool
        self.is_3D = val
        self.patch_size_Z.setVisible(self.is_3D)

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

        if path is not None and path != '':
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

    def _update_epochs(self, value=None):
        """
        Update the epoch progress bar following a change of total number of epochs.
        :return:
        """
        if self.state == State.IDLE:
            self.n_epochs = self.get_n_epochs()
            self.pb_epochs.setValue(0)
            self.pb_epochs.setMaximum(self.get_n_epochs())
            self.pb_epochs.setFormat(f'Epoch ?/{self.get_n_epochs()}')
        elif value is not None:
            # TODO not clean and hacky...
            # should only happen when retraining
            self.n_epochs = value
            self.pb_epochs.setValue(0)
            self.pb_epochs.setMaximum(value)
            self.pb_epochs.setFormat(f'Epoch ?/{value}')

    def get_n_epochs(self):
        return self.n_epochs_spin.value()

    def _update_steps(self, value=None):
        """
        Update the step progress bar following a change of total number of steps.
        :return:
        """
        if self.state == State.IDLE:
            self.n_steps = self.n_steps_spin.value()
            self.pb_steps.setValue(0)
            self.pb_steps.setMaximum(self.n_steps_spin.value())
            self.pb_steps.setFormat(f'Step ?/{self.n_steps_spin.value()}')
        elif value is not None:
            self.n_steps = value
            self.pb_steps.setValue(0)
            self.pb_steps.setMaximum(value)
            self.pb_steps.setFormat(f'Step ?/{value}')

    def _update_all(self, updates):
        """
        Update the UI following an update event from the training worker.
        :param updates:
        :return:
        """
        if self.state == State.RUNNING:
            if UpdateType.EPOCH in updates:
                val = updates[UpdateType.EPOCH]
                self.pb_epochs.setValue(val)
                self.pb_epochs.setFormat(f'Epoch {val}/{self.n_epochs}')

            if UpdateType.BATCH in updates:
                val = updates[UpdateType.BATCH]
                self.pb_steps.setValue(val)
                self.pb_steps.setFormat(f'Step {val}/{self.n_steps}')

            if UpdateType.LOSS in updates:
                self.plot.update_plot(*updates[UpdateType.LOSS])

            if UpdateType.RETRAIN in updates:
                self._update_epochs(value=updates[UpdateType.RETRAIN])

    def _save_model(self):
        """
        Export the model.
        :return:
        """
        if self.state == State.IDLE:
            if self.model:
                where = Path(QFileDialog.getSaveFileName(caption='Save model')[0])
                export_type = self.save_choice.currentText()

                # save
                parameters = {
                    'export_type': export_type,
                    'model': self.model,
                    'axes': self.new_axes,
                    'input_path': self.inputs,
                    'output_path': self.outputs,
                    'tf_version': self.tf_version
                }
                save_model(where, **parameters)

    def _training_expert_setter(self):
        if self.expert_settings is None:
            self.expert_settings = TrainingSettingsWidget(self)
        self.expert_settings.show()


if __name__ == "__main__":
    from napari_denoiseg._sample_data import denoiseg_data_2D_n20, denoiseg_data_3D_n20

    dims = '2D'  # '3D'
    if dims == '3D':
        data = denoiseg_data_3D_n20()
    else:
        data = denoiseg_data_2D_n20()

    # create a Viewer
    viewer = napari.Viewer()

    # add images
    viewer.add_image(data[0][0][:300], name=data[0][1]['name'])
    viewer.add_labels(data[1][0][:50], name=data[1][1]['name'])

    # add our plugin
    viewer.window.add_dock_widget(TrainingWidgetWrapper(viewer))

    napari.run()
