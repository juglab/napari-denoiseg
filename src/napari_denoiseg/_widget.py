"""
"""
import napari
from tensorflow.keras.callbacks import Callback
from napari.qt.threading import thread_worker
from magicgui import magic_factory
from magicgui.widgets import ProgressBar
from queue import Queue
import numpy as np
from utils import tb_plot_widget


# should probably refactor this, what would be a pythonic way?
# java way would be to create an object with member indicating
# whether the update is an "epoch end" update. This would allow
# more elegant check than the size of the list.
class Updater(Callback):
    def __init__(self):
        self.queue = Queue(10)
        self.epoch = 0
        self.batch = 0
        self.training_running = True

    def is_running(self):
        return self.training_running

    def training_done(self):
        self.training_running = False

    def update_epoch(self, epoch, train_loss, val_loss):
        self.epoch = epoch
        self.queue.put((self.epoch, self.batch, train_loss, val_loss))

    def update_batch(self, batch):
        self.batch = batch
        self.queue.put((self.epoch, self.batch))

    def get_queue_len(self):
        return self.queue.qsize()

    def on_epoch_end(self, epoch, logs=None):
        self.update_epoch(epoch, logs['loss'], logs['val_loss'])

    def on_train_batch_end(self, batch, logs=None):
        self.update_batch(batch)

    def on_train_end(self, logs=None):
        self.training_done()

# here the call to the progress bar is WRONG, because it creates ProgressBar with ProgressBar value
@magic_factory(perc_train_labels={"widget_type": "FloatSlider", "min": 0.1, "max": 1., "step": 0.05, 'value': 0.6},
               n_epochs={"widget_type": "SpinBox", "step": 1, 'value': 20},  # 10
               n_steps={"widget_type": "SpinBox", "step": 1, 'value': 4},  # 400
               batch_size={"widget_type": "Slider", "min": 8, "max": 512, "step": 16, 'value': 8},  # 64
               epoch_prog={'visible': True, 'min': 0, 'max': 100, 'step': 1, 'value': 0, 'label': 'epochs'},
               step_prog={'visible': True, 'min': 0, 'max': 100, 'step': 1, 'value': 0, 'label': 'steps'})
def denoiseg_widget(napari_viewer: 'napari.viewer.Viewer',
                    data: 'napari.layers.Image',
                    ground_truth: 'napari.layers.Labels',
                    perc_train_labels: float,
                    n_epochs: int,
                    n_steps: int,
                    batch_size: int,
                    epoch_prog: ProgressBar,
                    step_prog: ProgressBar):
    def started():
        pass

    def finished():
        pass

    def update_progress(update):
        epoch_prog.native.setValue(update[0])
        epoch_prog.native.setFormat(update[1])
        step_prog.native.setValue(update[2])
        step_prog.native.setFormat(update[3])

        if update[4][1]:
            plot_graph.update_plot(*update[4])

    @thread_worker(connect={'yielded': update_progress, 'started': started, 'finished': finished})
    def process(config, X_train, Y_train, X_val, Y_val):
        import threading

        # create updater
        denoiseg_updater = Updater()

        train_args = prepare_training(config, X_train, Y_train, X_val, Y_val, denoiseg_updater)

        training = threading.Thread(target=train, args=train_args)
        training.start()

        while True:
            el = denoiseg_updater.queue.get(True)
            e, s = el[0] + 1, el[1] + 1  # 1-indexed
            perc_e = int(100 * e / n_epochs + 0.5)
            perc_s = int(100 * s / n_steps + 0.5)

            if len(el) > 2:
                tl, vl = el[2], el[3]
            else:
                tl, vl = None, None

            yield perc_e, f'Epoch {e}/{n_epochs}', perc_s, f'Step {s}/{n_steps}', (e, tl, vl)

            if e == n_epochs and s == n_steps:
                print('Training done')
                break

    # split train and val
    print(f'Data shape: data {data.data.shape}, gt {ground_truth.data.shape}')
    X_t, Y_t, X_v, Y_v = prepare_data(data.data, ground_truth.data, perc_train_labels)
    print(f'Data shape: X {X_t.shape}, Y {Y_t.shape}, X_val {X_v.shape} and Y {Y_v.shape}')

    # create DenoiSeg configuration
    denoiseg_conf = generate_config(X_t, n_epochs, n_steps, batch_size)

    # create plot_graph: note clicking on run will create a new one
    plot_graph = tb_plot_widget.tb_plot_widget
    napari_viewer.window.add_dock_widget(plot_graph)

    # to stop the tensorboard, but this yields a warning because we access a hidden member
    # I keep here for reference since I haven't found a good way to stop the tb (they have no closing API)
    #napari_viewer.window.qt_viewer.destroyed.connect(plot_graph.stop_tb)

    # start process
    process(denoiseg_conf, X_t, Y_t, X_v, Y_v)


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
    n_labels = int(0.5 + perc_labels * gt.data.shape[0])
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

    return X, Y, X_val, Y_val


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
    history = model.keras_model.fit(training_data, validation_data=(validation_X, validation_Y),
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

    return history


if __name__ == "__main__":
    with napari.gui_qt():
        noise_level = 'n0'

        # Loading of the training images
        train_data = np.load('data/DSB2018_{}/train/train_data.npz'.format(noise_level))
        images = train_data['X_train'].astype(np.float32)[0:30, :, :]
        labels = train_data['Y_train'].astype(np.int32)[0:16, :, :]

        # create a Viewer and add an image here
        viewer = napari.Viewer()

        # add images
        viewer.add_image(images)
        viewer.add_labels(labels)

        # custom code to add data here
        viewer.window.add_dock_widget(denoiseg_widget())

        napari.run()
