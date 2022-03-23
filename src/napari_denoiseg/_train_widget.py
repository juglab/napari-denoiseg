"""
"""
import napari
from tensorflow.keras.callbacks import Callback
from napari.qt.threading import thread_worker
from magicgui import magic_factory
from magicgui.widgets import create_widget, Container
from queue import Queue
import numpy as np
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
    QLabel
)
from enum import Enum
from napari_denoiseg._tbplot_widget import TBPlotWidget


class State(Enum):
    IDLE = 0
    RUNNING = 1


class Updates(Enum):
    EPOCH = 'epoch'
    BATCH = 'batch'
    LOSS = 'loss'
    DONE = 'done'


class SaveMode(Enum):
    TF = 'TensorFlow'
    MODELZOO = 'Bioimage.io'

    @classmethod
    def list(cls):
        return list(map(lambda c: c.value, cls))


class Updater(Callback):
    def __init__(self):
        self.queue = Queue(10)
        self.epoch = 0
        self.batch = 0

    def on_epoch_begin(self, epoch, logs=None):
        self.epoch = epoch
        self.queue.put({Updates.EPOCH: self.epoch + 1})

    def on_epoch_end(self, epoch, logs=None):
        self.queue.put({Updates.LOSS: (self.epoch, logs['loss'], logs['val_loss'])})

    def on_train_batch_begin(self, batch, logs=None):
        self.batch = batch
        self.queue.put({Updates.BATCH: self.batch + 1})

    def on_train_end(self, logs=None):
        self.queue.put(Updates.DONE)

    def stop_training(self):
        self.model.stop_training = True


def create_choice_widget(napari_viewer):
    def layer_choice_widget(np_viewer, annotation, **kwargs):
        widget = create_widget(annotation=annotation, **kwargs)
        widget.reset_choices()
        np_viewer.layers.events.inserted.connect(widget.reset_choices)
        np_viewer.layers.events.removed.connect(widget.reset_choices)
        return widget

    img = layer_choice_widget(napari_viewer, annotation=napari.layers.Image, name="Images")
    lbl = layer_choice_widget(napari_viewer, annotation=napari.layers.Labels, name="Masks")

    return Container(widgets=[img, lbl])


@magic_factory(auto_call=True,
               labels=False,
               slider={"widget_type": "Slider", "min": 0, "max": 100, "step": 5, 'value': 60})
def get_perc_train_slider(slider: int):
    pass


@magic_factory(auto_call=True,
               labels=False,
               slider={"widget_type": "Slider", "min": 8, "max": 512, "step": 16, 'value': 8})
def get_batch_size_slider(slider: int):
    pass


class TrainWidget(QWidget):
    def __init__(self, napari_viewer):
        super().__init__()

        self.state = State.IDLE
        self.viewer = napari_viewer

        self.setLayout(QVBoxLayout())

        # layer choice widgets
        self.layer_choice = create_choice_widget(napari_viewer)
        self.images = self.layer_choice.Images
        self.labels = self.layer_choice.Masks
        self.layout().addWidget(self.layer_choice.native)

        # others
        self.perc_train_slider = get_perc_train_slider()

        self.n_epochs_spin = QSpinBox()
        self.n_epochs_spin.setMinimum(1)
        self.n_epochs_spin.setValue(2)
        self.n_epochs = self.n_epochs_spin.value()

        self.n_steps_spin = QSpinBox()
        self.n_steps_spin.setMinimum(1)
        self.n_steps_spin.setValue(10)
        self.n_steps = self.n_steps_spin.value()

        self.batch_size_slider = get_batch_size_slider()

        others = QWidget()
        formLayout = QFormLayout()
        formLayout.addRow('Train label %', self.perc_train_slider.native)
        formLayout.addRow('N epochs', self.n_epochs_spin)
        formLayout.addRow('N steps', self.n_steps_spin)
        formLayout.addRow('Batch size', self.batch_size_slider.native)
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
        self.train_button = QPushButton("Train", self)
        self.layout().addWidget(self.train_button)

        self.threshold_label = QLabel()
        self.threshold_label.setText("Best threshold: ?")
        self.layout().addWidget(self.threshold_label)

        # Save button
        save_widget = QWidget()
        save_widget.setLayout(QHBoxLayout())

        self.save_choice = QComboBox()
        self.save_choice.addItems(SaveMode.list())

        self.save_button = QPushButton("Save model", self)
        self.save_button.setEnabled(False)

        save_widget.layout().addWidget(self.save_button)
        save_widget.layout().addWidget(self.save_choice)
        self.layout().addWidget(save_widget)

        # plot widget
        self.plot = TBPlotWidget(300, 300)
        self.layout().addWidget(self.plot.native)

        # worker
        self.worker = None
        self.train_button.clicked.connect(self.start_training)

        # actions
        self.n_epochs_spin.valueChanged.connect(self.update_epochs)
        self.n_steps_spin.valueChanged.connect(self.update_steps)

        # this allows stopping the thread when the napari window is closed,
        # including reducing the risk that an update comes after closing the
        # window and appearing as a new Qt view. But the call to qt_viewer
        # will be deprecated. Hopefully until then an on_window_closing event
        # will be available.
        napari_viewer.window.qt_viewer.destroyed.connect(self.interrupt)

        # place-holder for the trained model
        self.model, self.X_val, self.threshold = None, None, None
        self.save_button.clicked.connect(self.save_model)

    def interrupt(self):
        self.worker.quit()

    def start_training(self):
        if self.state == State.IDLE:
            self.state = State.RUNNING

            self.plot.clear_plot()
            self.train_button.setText('Stop')

            self.threshold_label.setText("Best threshold: ?")

            self.save_button.setEnabled(False)

            self.worker = train_worker(self)
            self.worker.yielded.connect(lambda x: self.update_all(x))
            self.worker.returned.connect(self.done)
            self.worker.start()
        elif self.state == State.RUNNING:
            self.state = State.IDLE

    def done(self):
        self.state = State.IDLE
        self.train_button.setText('Train again')

        self.threshold_label.setText("Best threshold: " + str(self.threshold))

        self.save_button.setEnabled(True)

    def update_epochs(self):
        if self.state == State.IDLE:
            self.n_epochs = self.n_epochs_spin.value()
            self.pb_epochs.setValue(0)
            self.pb_epochs.setFormat(f'Epoch ?/{self.n_epochs_spin.value()}')

    def update_steps(self):
        if self.state == State.IDLE:
            self.n_steps = self.n_steps_spin.value()
            self.pb_steps.setValue(0)
            self.pb_steps.setFormat(f'Step ?/{self.n_steps_spin.value()}')

    def update_all(self, updates):
        if self.state == State.RUNNING:
            if Updates.EPOCH in updates:
                val = updates[Updates.EPOCH]
                e_perc = int(100 * updates[Updates.EPOCH] / self.n_epochs + 0.5)
                self.pb_epochs.setValue(e_perc)
                self.pb_epochs.setFormat(f'Epoch {val}/{self.n_epochs}')

            if Updates.BATCH in updates:
                val = updates[Updates.BATCH]
                s_perc = int(100 * val / self.n_steps + 0.5)
                self.pb_steps.setValue(s_perc)
                self.pb_steps.setFormat(f'Epoch {val}/{self.n_steps}')

            if Updates.LOSS in updates:
                self.plot.update_plot(*updates[Updates.LOSS])

    def save_model(self):
        if self.state == State.IDLE:
            if self.model:
                where = QFileDialog.getSaveFileName(caption='Save model')[0]

                export_type = self.save_choice.currentText()
                if SaveMode.MODELZOO.value == export_type:
                    self.model.export_TF(name='DenoiSeg',
                                         description='Trained DenoiSeg model',
                                         authors=["Tim-Oliver Buchholz", "Mangal Prakash", "Alexander Krull",
                                                  "Florian Jug"],
                                         test_img=self.X_val[0, ..., 0], axes='YX',
                                         patch_shape=(128, 128), fname=where + '.bioimage.io.zip')
                else:
                    self.model.keras_model.save_weights(where + '.h5')


@thread_worker(start_thread=False)
def train_worker(widget: TrainWidget):
    import threading
    from denoiseg.utils.compute_precision_threshold import measure_precision

    # get images and labels
    image_data = widget.images.value.data
    label_data = widget.labels.value.data

    # split train and val
    perc_labels = widget.perc_train_slider.slider.get_value()
    X_t, Y_t, X_v, Y_v, validation_x, validation_y = prepare_data(image_data, label_data, perc_labels)

    # create DenoiSeg configuration
    n_epochs = widget.n_epochs
    n_steps = widget.n_steps
    batch_size = widget.batch_size_slider.slider.get_value()
    denoiseg_conf = generate_config(X_t, n_epochs, n_steps, batch_size)

    # to stop the tensorboard, but this yields a warning because we access a hidden member
    # I keep here for reference since I haven't found a good way to stop the tb (they have no closing API)
    # napari_viewer.window.qt_viewer.destroyed.connect(plot_graph.stop_tb)

    # create updater
    denoiseg_updater = Updater()

    train_args = prepare_training(denoiseg_conf, X_t, Y_t, X_v, Y_v, denoiseg_updater)

    training = threading.Thread(target=train, args=train_args)
    training.start()

    # loop looking for update events
    while True:
        update = denoiseg_updater.queue.get(True)

        if Updates.DONE == update:
            break
        elif widget.state != State.RUNNING:
            denoiseg_updater.stop_training()
            yield Updates.DONE
            break
        else:
            yield update

    # training done, keep model in memory
    widget.model, widget.X_val = train_args[0], train_args[2]

    # threshold validation data to estimate the best threshold
    threshold, val_score = widget.model.optimize_thresholds(validation_x,
                                                            validation_y,
                                                            measure=measure_precision())

    print("The highest score of {} is achieved with threshold = {}.".format(np.round(val_score, 3), threshold))
    widget.threshold = threshold


# refactor with prepare_training
def prepare_data(data, gt, perc_labels):
    from denoiseg.utils.misc_utils import augment_data
    from denoiseg.utils.seg_utils import convert_to_oneHot

    def zero_sum(im):
        im_reshaped = im.reshape(im.shape[0], -1)

        return np.where(np.sum(im_reshaped != 0, axis=1) != 0)[0]

    def list_diff(l1, l2):
        return list(set(l1) - set(l2))

    def create_train_set(x, y, ind_exclude):
        masks = np.zeros(x.shape)
        masks[0:y.shape[0], 0:y.shape[1], 0:y.shape[2]] = y  # there's probably a more elegant way

        return augment_data(np.delete(x, ind_exclude, axis=0), np.delete(masks, ind_exclude, axis=0))

    def create_val_set(x, y, ind_include):
        return np.take(x, ind_include, axis=0), np.take(y, ind_include, axis=0)

    # currently: no shuffling. we detect the non-empty labeled frames and split them. The validation set is
    # entirely constituted of the frames corresponding to the split labeled frames, the training set is all the
    # remaining images. The training gt is then the remaining labeled frames and empty frames.

    # get indices of labeled frames
    dec_perc_labels = 0.01 * perc_labels
    n_labels = int(0.5 + dec_perc_labels * gt.data.shape[0])
    ind = zero_sum(gt)
    assert n_labels < len(ind)

    # split labeled frames between train and val sets
    ind_train = np.random.choice(ind, size=n_labels, replace=False).tolist()
    ind_val = list_diff(ind, ind_train)
    assert len(ind_train) + len(ind_val) == len(ind)

    # create train and val sets
    x_train, y_train = create_train_set(data, gt, ind_val)
    x_val, y_val = create_val_set(data, gt, ind_val)

    # add channel dim and one-hot encoding
    X = x_train[..., np.newaxis]
    Y = convert_to_oneHot(y_train)
    X_val = x_val[..., np.newaxis]
    Y_val = convert_to_oneHot(y_val)

    return X, Y, X_val, Y_val, x_val, y_val


def generate_config(X, n_epochs, n_steps, batch_size):
    from denoiseg.models import DenoiSegConfig

    conf = DenoiSegConfig(X, unet_kern_size=3, n_channel_out=4, relative_weights=[1.0, 1.0, 5.0],
                          train_steps_per_epoch=n_steps, train_epochs=n_epochs,
                          batch_norm=True, train_batch_size=batch_size,
                          unet_n_first=32, unet_n_depth=4, denoiseg_alpha=0.5, train_tensorboard=True)

    return conf


def prepare_training(conf, X_train, Y_train, X_val, Y_val, updater):
    from datetime import date
    import tensorflow as tf
    import warnings
    from denoiseg.models import DenoiSeg
    from csbdeep.utils import axes_check_and_normalize, axes_dict
    from denoiseg.internals.DenoiSeg_DataWrapper import DenoiSeg_DataWrapper
    from n2v.utils import n2v_utils
    from n2v.utils.n2v_utils import pm_uniform_withCP

    today = date.today().strftime("%b-%d-%Y")

    model_name = 'DenoiSeg_' + today
    basedir = 'models'

    if tf.config.list_physical_devices('GPU'):
        tf.config.experimental.set_memory_growth(tf.config.list_physical_devices('GPU')[0], True)
    model = DenoiSeg(conf, model_name, basedir)

    # training loop
    n_train, n_val = len(X_train), len(X_val)
    frac_val = (1.0 * n_val) / (n_train + n_val)
    frac_warn = 0.05
    if frac_val < frac_warn:
        warnings.warn("small number of validation images (only %.05f%% of all images)" % (100 * frac_val))
    axes = axes_check_and_normalize('S' + model.config.axes, X_train.ndim)
    ax = axes_dict(axes)
    div_by = 2 ** model.config.unet_n_depth
    axes_relevant = ''.join(a for a in 'XYZT' if a in axes)
    val_num_pix = 1
    train_num_pix = 1
    val_patch_shape = ()
    for a in axes_relevant:
        n = X_train.shape[ax[a]]
        val_num_pix *= X_val.shape[ax[a]]
        train_num_pix *= X_train.shape[ax[a]]
        val_patch_shape += tuple([X_val.shape[ax[a]]])
        if n % div_by != 0:
            raise ValueError(
                "training images must be evenly divisible by %d along axes %s"
                " (axis %s has incompatible size %d)" % (div_by, axes_relevant, a, n)
            )

    epochs = model.config.train_epochs
    steps_per_epoch = model.config.train_steps_per_epoch

    model.prepare_for_training()

    manipulator = eval(
        'pm_{0}({1})'.format(model.config.n2v_manipulator, str(model.config.n2v_neighborhood_radius)))

    means = np.array([float(mean) for mean in model.config.means], ndmin=len(X_train.shape), dtype=np.float32)
    stds = np.array([float(std) for std in model.config.stds], ndmin=len(X_train.shape), dtype=np.float32)

    X = model.__normalize__(X_train, means, stds)
    validation_X = model.__normalize__(X_val, means, stds)

    # Here we prepare the Noise2Void data. Our input is the noisy data X and as target we take X concatenated with
    # a masking channel. The N2V_DataWrapper will take care of the pixel masking and manipulating.
    training_data = DenoiSeg_DataWrapper(X=X,
                                         n2v_Y=np.concatenate((X, np.zeros(X.shape, dtype=X.dtype)),
                                                              axis=axes.index('C')),
                                         seg_Y=Y_train,
                                         batch_size=model.config.train_batch_size,
                                         perc_pix=model.config.n2v_perc_pix,
                                         shape=model.config.n2v_patch_shape,
                                         value_manipulation=manipulator)

    # validation_Y is also validation_X plus a concatenated masking channel.
    # To speed things up, we precompute the masking vo the validation data.
    validation_Y = np.concatenate((validation_X, np.zeros(validation_X.shape, dtype=validation_X.dtype)),
                                  axis=axes.index('C'))
    n2v_utils.manipulate_val_data(validation_X, validation_Y,
                                  perc_pix=model.config.n2v_perc_pix,
                                  shape=val_patch_shape,
                                  value_manipulation=manipulator)

    validation_Y = np.concatenate((validation_Y, Y_val), axis=-1)

    # add callbacks
    model.callbacks.append(updater)

    return model, training_data, validation_X, validation_Y, epochs, steps_per_epoch


def train(model, training_data, validation_X, validation_Y, epochs, steps_per_epoch):
    model.keras_model.fit(training_data, validation_data=(validation_X, validation_Y),
                          epochs=epochs, steps_per_epoch=steps_per_epoch,
                          callbacks=model.callbacks, verbose=1)

    if model.basedir is not None:
        model.keras_model.save_weights(str(model.logdir / 'weights_last.h5'))

        if model.config.train_checkpoint is not None:
            print()
            model._find_and_load_weights(model.config.train_checkpoint)
            try:
                # remove temporary weights
                (model.logdir / 'weights_now.h5').unlink()
            except FileNotFoundError:
                pass


if __name__ == "__main__":
    with napari.gui_qt():
        noise_level = 'n0'

        # Loading of the training images
        train_data = np.load('data/DSB2018_{}/train/train_data.npz'.format(noise_level))
        images = train_data['X_train'].astype(np.float32)[0:30, :, :]
        labels = train_data['Y_train'].astype(np.int32)[0:16, :, :]

        # create a Viewer and add an image here
        viewer = napari.Viewer()

        # custom code to add data here
        viewer.window.add_dock_widget(TrainWidget(viewer))

        # add images
        viewer.add_image(images, name='Images')
        viewer.add_labels(labels, name='Labels')

        napari.run()
