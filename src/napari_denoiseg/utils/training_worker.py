from queue import Queue
import numpy as np

from tensorflow.keras.callbacks import Callback

from napari.qt.threading import thread_worker
from napari_denoiseg.utils import State, UpdateType


class TrainingCallback(Callback):
    def __init__(self):
        self.queue = Queue(10)
        self.epoch = 0
        self.batch = 0

    def on_epoch_begin(self, epoch, logs=None):
        self.epoch = epoch
        self.queue.put({UpdateType.EPOCH: self.epoch + 1})

    def on_epoch_end(self, epoch, logs=None):
        if logs:
            self.queue.put({UpdateType.LOSS: (self.epoch, logs['loss'], logs['val_loss'])})

    def on_train_batch_begin(self, batch, logs=None):
        self.batch = batch
        self.queue.put({UpdateType.BATCH: self.batch + 1})

    def on_train_end(self, logs=None):
        self.queue.put(UpdateType.DONE)

    def stop_training(self):
        self.model.stop_training = True


@thread_worker(start_thread=False)
def training_worker(widget, pretrained_model=None):
    import os
    import threading
    from denoiseg.utils.compute_precision_threshold import measure_precision
    from napari_denoiseg.utils import UpdateType, generate_config

    # get images and labels
    X_t, Y_t, X_v, Y_v, validation_x, validation_y = load_images(widget)

    # create DenoiSeg configuration
    n_epochs = widget.n_epochs
    n_steps = widget.n_steps
    batch_size = widget.batch_size_spin.value()
    patch_shape = widget.patch_size_spin.value()
    denoiseg_conf = generate_config(X_t, n_epochs, n_steps, batch_size, patch_shape)

    # prepare training
    args = (denoiseg_conf, X_t, Y_t, X_v, Y_v, pretrained_model)
    train_args, denoiseg_updater, widget.tf_version = prepare_training(*args)

    # start training
    training = threading.Thread(target=train, args=train_args)
    training.start()

    # loop looking for update events using the updater queue
    while True:
        update = denoiseg_updater.queue.get(True)

        if UpdateType.DONE == update:
            break
        elif widget.state != State.RUNNING:
            denoiseg_updater.stop_training()
            yield UpdateType.DONE
            break
        else:
            yield update

    # training done, keep model in memory
    widget.model = train_args[0]

    # threshold validation data to estimate the best threshold
    # TODO would be better to use the same code base than in the optimizer_worker
    widget.threshold, val_score = widget.model.optimize_thresholds(validation_x,
                                                                   validation_y,
                                                                   measure=measure_precision())

    # save input/output for bioimage.io
    widget.inputs = os.path.join(widget.model.basedir, 'inputs.npy')
    widget.outputs = os.path.join(widget.model.basedir, 'outputs.npy')
    np.save(widget.inputs, validation_x[np.newaxis, 0, ..., np.newaxis])
    np.save(widget.outputs, widget.model.predict(validation_x[np.newaxis, 0, ..., np.newaxis], axes='SYXC'))


def load_images(widget):
    # get images and labels
    if widget.load_from_disk:
        # folders
        path_train_X = widget.train_images_folder.get_folder()
        path_train_Y = widget.train_labels_folder.get_folder()
        path_val_X = widget.val_images_folder.get_folder()
        path_val_Y = widget.val_labels_folder.get_folder()

        X_t, Y_t, X_v, Y_v, validation_x, validation_y = prepare_data_disk(path_train_X,
                                                                           path_train_Y,
                                                                           path_val_X,
                                                                           path_val_Y)
    else:
        # get layers
        image_data = widget.images.value.data
        label_data = widget.labels.value.data

        # split train and val
        perc_labels = widget.perc_train_slider.slider.get_value()
        X_t, Y_t, X_v, Y_v, validation_x, validation_y = prepare_data_layers(image_data, label_data, perc_labels)
    return X_t, Y_t, X_v, Y_v, validation_x, validation_y


def prepare_data_disk(train_source, train_target, val_source, val_target):
    from denoiseg.utils.misc_utils import augment_data
    from denoiseg.utils.seg_utils import convert_to_oneHot
    from napari_denoiseg.utils import load_pairs_from_disk

    # load train data
    _x_train, _y_train = load_pairs_from_disk(train_source, train_target, check_exists=False)

    # apply augmentation
    x_train, y_train = augment_data(_x_train, _x_train)

    # load val data
    x_val, y_val = load_pairs_from_disk(val_source, val_target)  # val sets without one-hot encoding

    # add channel dim and one-hot encoding
    X = x_train[..., np.newaxis]
    Y = convert_to_oneHot(y_train)

    X_val = x_val[..., np.newaxis]
    Y_val = convert_to_oneHot(y_val)

    return X, Y, X_val, Y_val, x_val, y_val


def prepare_data_layers(raw, gt, perc_labels):
    """

    :param raw:
    :param gt:
    :param perc_labels:
    :return:
    """
    from denoiseg.utils.misc_utils import augment_data
    from denoiseg.utils.seg_utils import convert_to_oneHot

    def zero_sum(im):
        """
        Detect empty slices.
        :param im: Image stack
        :return:
        """
        im_reshaped = im.reshape(im.shape[0], -1)

        return np.where(np.sum(im_reshaped != 0, axis=1) != 0)[0]

    def list_diff(l1, l2):
        """
        Return the difference of two lists.
        :param l1:
        :param l2:
        :return: list of elements in l1 that are not in l2.
        """
        return list(set(l1) - set(l2))

    def create_train_set(x, y, ind_exclude):
        masks = np.zeros(x.shape)
        masks[0:y.shape[0], 0:y.shape[1], 0:y.shape[2]] = y  # there's probably a more elegant way

        return augment_data(np.delete(x, ind_exclude, axis=0), np.delete(masks, ind_exclude, axis=0))

    def create_val_set(x, y, ind_include):
        return np.take(x, ind_include, axis=0), np.take(y, ind_include, axis=0)

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
    x_train, y_train = create_train_set(raw, gt, ind_val)
    x_val, y_val = create_val_set(raw, gt, ind_val)  # val sets without one-hot encoding

    # add channel dim and one-hot encoding
    X = x_train[..., np.newaxis]
    Y = convert_to_oneHot(y_train)
    X_val = x_val[..., np.newaxis]
    Y_val = convert_to_oneHot(y_val)

    return X, Y, X_val, Y_val, x_val, y_val


def sanity_check_validation_fraction(X_train, X_val):
    import warnings
    n_train, n_val = len(X_train), len(X_val)
    frac_val = (1.0 * n_val) / (n_train + n_val)
    frac_warn = 0.05
    if frac_val < frac_warn:
        warnings.warn("small number of validation images (only %.05f%% of all images)" % (100 * frac_val))


def sanity_check_training_size(X_train, model, axes):
    from csbdeep.utils import axes_dict

    ax = axes_dict(axes)
    axes_relevant = ''.join(a for a in 'XYZT' if a in axes)
    div_by = 2 ** model.config.unet_n_depth
    for a in axes_relevant:
        n = X_train.shape[ax[a]]
        if n % div_by != 0:
            raise ValueError(
                "training images must be evenly divisible by %d along axes %s"
                " (axis %s has incompatible size %d)" % (div_by, axes_relevant, a, n)
            )


def get_validation_patch_shape(X_val, axes):
    from csbdeep.utils import axes_dict

    ax = axes_dict(axes)
    axes_relevant = ''.join(a for a in 'XYZT' if a in axes)
    val_patch_shape = ()
    for a in axes_relevant:
        val_patch_shape += tuple([X_val.shape[ax[a]]])

    return val_patch_shape


def normalize_images(model, X_train, X_val):
    means = np.array([float(mean) for mean in model.config.means], ndmin=len(X_train.shape), dtype=np.float32)
    stds = np.array([float(std) for std in model.config.stds], ndmin=len(X_train.shape), dtype=np.float32)

    X = model.__normalize__(X_train, means, stds)
    validation_X = model.__normalize__(X_val, means, stds)

    return X, validation_X


def prepare_training(conf, X_train, Y_train, X_val, Y_val, pretrained_model=None):
    from datetime import date
    import tensorflow as tf
    from denoiseg.models import DenoiSeg
    from csbdeep.utils import axes_check_and_normalize
    from denoiseg.internals.DenoiSeg_DataWrapper import DenoiSeg_DataWrapper
    from n2v.utils import n2v_utils
    from n2v.utils.n2v_utils import pm_uniform_withCP

    # create model
    today = date.today().strftime("%b-%d-%Y")
    model_name = 'DenoiSeg_' + today
    basedir = 'models'
    model = DenoiSeg(conf, model_name, basedir)

    # if tf.config.list_physical_devices('GPU'):
    #    tf.config.experimental.set_memory_growth(tf.config.list_physical_devices('GPU')[0], True)

    # if we continue training, transfer the weights
    # TODO: currently not reusing the same model in case the configuration changed (n epochs, data etc.)
    if pretrained_model:
        model.keras_model.set_weights(pretrained_model.keras_model.get_weights())

    # normalize axes
    axes = axes_check_and_normalize('S' + model.config.axes, X_train.ndim)

    # run some sanity check
    sanity_check_validation_fraction(X_train, X_val)
    sanity_check_training_size(X_train, model, axes)

    # compute validation patch shape
    val_patch_shape = get_validation_patch_shape(X_val, axes)

    # prepare model for training
    model.prepare_for_training()

    manipulator = eval(
        'pm_{0}({1})'.format(model.config.n2v_manipulator, str(model.config.n2v_neighborhood_radius)))

    # normalize images
    X, validation_X = normalize_images(model, X_train, X_val)

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
    updater = TrainingCallback()
    model.callbacks.append(updater)

    # training parameters
    epochs = model.config.train_epochs
    steps_per_epoch = model.config.train_steps_per_epoch
    train_params = (model, training_data, validation_X, validation_Y, epochs, steps_per_epoch)

    return train_params, updater, tf.__version__


def train(model, training_data, validation_X, validation_Y, epochs, steps_per_epoch):
    # start training
    model.keras_model.fit(training_data, validation_data=(validation_X, validation_Y),
                          epochs=epochs, steps_per_epoch=steps_per_epoch,
                          callbacks=model.callbacks, verbose=1)

    # save last weights
    if model.basedir is not None:
        model.keras_model.save_weights(str(model.logdir / 'weights_last.h5'))
