from queue import Queue

import napari_svg.layer_to_xml
import numpy as np

from tensorflow.keras.callbacks import Callback

from napari.qt.threading import thread_worker
from denoiseg.utils.seg_utils import convert_to_oneHot
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
    X_train, Y_train, X_val, Y_val_onehot, Y_val, widget.new_axes = load_images(widget)

    # create DenoiSeg configuration
    n_epochs = widget.n_epochs
    n_steps = widget.n_steps
    batch_size = widget.batch_size_spin.value()
    patch_shape = widget.patch_size_spin.value()
    denoiseg_conf = generate_config(X_train, n_epochs, n_steps, batch_size, patch_shape)

    # prepare training
    args = (denoiseg_conf, X_train, Y_train, X_val, Y_val_onehot, pretrained_model)
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
    widget.threshold, val_score = widget.model.optimize_thresholds(X_val[..., 0],
                                                                   Y_val,
                                                                   measure=measure_precision())

    # save input/output for bioimage.io
    widget.inputs = os.path.join(widget.model.basedir, 'inputs.npy')
    widget.outputs = os.path.join(widget.model.basedir, 'outputs.npy')
    np.save(widget.inputs, X_val[..., 0][np.newaxis, 0, ..., np.newaxis])
    np.save(widget.outputs, widget.model.predict(X_val[..., 0][np.newaxis, 0, ..., np.newaxis],
                                                 axes=widget.new_axes))


def load_images(widget):
    """

    :param widget:
    :return:
    """
    # TODO make clearer what objects are returned

    # get images and labels
    if widget.load_from_disk:  # from local folders
        path_train_X = widget.train_images_folder.get_folder()
        path_train_Y = widget.train_labels_folder.get_folder()
        path_val_X = widget.val_images_folder.get_folder()
        path_val_Y = widget.val_labels_folder.get_folder()

        return prepare_data_disk(path_train_X, path_train_Y, path_val_X, path_val_Y, widget.axes)

    else:  # from layers
        image_data = widget.images.value.data
        label_data = widget.labels.value.data

        # split train and val
        perc_labels = widget.perc_train_slider.slider.get_value()

        return prepare_data_layers(image_data, label_data, perc_labels, widget.axes)


def reshape_data(x, y, axes: str):
    """
    Reshape the data to 'SZXYC' depending on the available `axes`. If a T dimension is present, the different time
    points are considered independent and stacked along the S dimension.

    Note that if C dimension is present, y will have a singleton dimension

    :param x: Raw data.
    :param y: Ground-truth data.
    :param axes: Current axes order of X
    :return: Reshaped x, reshaped y, new axes order
    """
    ref_axes = 'TSZYXC'

    # sanity checks TODO: raise error rather than assert?
    assert 'X' in axes and 'Y' in axes, 'X or Y dimension missing in axes.'

    if 'C' in axes:
        assert len(axes) == len(x.shape) == len(y.shape) + 1
    else:
        assert len(axes) == len(x.shape) == len(y.shape)

    assert len(list_diff(list(axes), list(ref_axes))) == 0  # all axes are part of ref_axes

    # if S is not in the list of axes, then add a singleton S
    if 'S' not in axes:
        _axes = 'S' + axes
        _x = x[np.newaxis, ...]
        _y = y[np.newaxis, ...]
    else:
        _axes = axes
        _x = x
        _y = y

    # build indices look-up table: indices of each axe in `axes`
    indices = [_axes.find(k) for k in ref_axes]

    # remove all non-existing axes (index == -1)
    indices = tuple(filter(lambda k: k != -1, indices))
    new_axes = [_axes[ind] for ind in indices]
    new_x_shape = tuple([_x.shape[ind] for ind in indices])
    new_y_shape = tuple([_y.shape[ind] for ind in indices])

    if 'C' in _axes:  # Y does not have a C dimension
        new_y_shape = new_y_shape[:-1]

    # remove T if necessary
    if 'T' in _axes:
        new_x_shape = (-1,) + new_x_shape[2:]  # remove T and S
        new_y_shape = (-1,) + new_y_shape[2:]
        new_axes.pop(0)

    # reshape
    _x = _x.reshape(new_x_shape)
    _y = _y.reshape(new_y_shape)

    # add channel
    if 'C' not in new_axes:
        _x = _x[..., np.newaxis]
        new_axes.append('C')

    return _x, _y, ''.join(new_axes)


def augment_data(array, axes: str):
    """
    Augments the data 8-fold by 90 degree rotations and flipping.

    Takes a dimension S.
    """
    # Adapted from DenoiSeg, in order to work with the following order `SZYXC`
    ind_x = axes.find('X')
    ind_y = axes.find('Y')

    # rotations
    _x = array.copy()
    X_rot = [np.rot90(_x, i, (ind_y, ind_x)) for i in range(4)]
    X_rot = np.concatenate(X_rot, axis=0)

    # flip
    X_flip = np.flip(X_rot, axis=ind_y)

    return np.concatenate([X_rot, X_flip], axis=0)


def load_data_from_disk(source, target, axes, augmentation=False, check_exists=True):
    """
    Load pairs of raw and label images (*.tif) from the disk.

    Accepted dimensions are 'SZYXC'. Different time points will be considered independent and added to the S dimension.

    The `check_exist` parameter allow replacing non-existent label images by empty images if it is set to `False`.

    `augmentation` yields an 8-fold augmentation for both source and target.

    This method returns a tuple (X, Y, X_val, Y_val, x_val, y_val, new_axes), where:
    X: raw images with dimension SYXC or SZYXC, with S(ample) and C(channel) (channel can be singleton dimension)
    Y: label images with dimension SYXC or SZYXC, with dimension C of length 3 (one-hot encoding)
    x: raw images without singleton C(hannel) dimension (if applicable)
    y: labels without one-hot encoding
    new_axes: new axes order

    :param source: Path to the folder containing the training images.
    :param target: Path to the folder containing the training labels.
    :param axes: Axes order
    :param augmentation: Augment data 8-fold if True
    :param check_exists: Check if source images have a target if True. Else it will load an empty image instead.
    :return:
    """
    from denoiseg.utils.seg_utils import convert_to_oneHot
    from napari_denoiseg.utils import load_pairs_from_disk

    # load train data
    _x, _y, n = load_pairs_from_disk(source, target, axes, check_exists=check_exists)

    # reshape data
    if n > 1:  # if multiple sample
        _axes = 'S' + axes
    else:
        _axes = axes
    _x, _y, new_axes = reshape_data(_x, _y, _axes)

    # apply augmentation, XY dimensions must be equal (dim(X) == dim(Y))
    if augmentation:
        _x = augment_data(_x, new_axes)
        _y = augment_data(_y, new_axes)
        print('Raw image size after augmentation', _x.shape)
        print('Mask size after augmentation', _y.shape)

    # add channel dim and one-hot encoding
    Y = convert_to_oneHot(_y)

    return _x, Y, _y, new_axes


def prepare_data_disk(path_train_X, path_train_Y, path_val_X, path_val_Y, axes):
    (X, Y, _, new_axes) = load_data_from_disk(path_train_X, path_train_Y, axes, True, False)
    (X_val, Y_val, y_val, _) = load_data_from_disk(path_val_X, path_val_Y, axes)

    return X, Y, X_val, Y_val, y_val, new_axes


def non_zero_sum(im):
    """
    Detect empty slices along the S dim.

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


def create_train_set(x, y, ind_exclude, axes):
    """

    :param axes:
    :param x:
    :param y:
    :param ind_exclude:
    :return:
    """
    # Different between x and y shapes:
    # if there are channels in x, there should be none in y
    # along the S dimension, x can be larger than y

    masks = np.zeros(x.shape[:-1])  # Y does not have channels dim before one hot encoding
    masks[:y.shape[0], ...] = y

    x_aug = augment_data(np.delete(x, ind_exclude, axis=0), axes)
    y_aug = augment_data(np.delete(masks, ind_exclude, axis=0), axes)

    return x_aug, y_aug


def create_val_set(x, y, ind_include):
    """

    :param x:
    :param y:
    :param ind_include:
    :return:
    """
    return np.take(x, ind_include, axis=0), np.take(y, ind_include, axis=0)


def prepare_data_layers(raw, gt, perc_labels, axes):
    """

    :param raw:
    :param gt:
    :param perc_labels:
    :param axes
    :return:
    """

    # reshape data
    _x, _y, new_axes = reshape_data(raw, gt, axes)
    # TODO: check that dim(X) and dim(Y) are equal for the augmentation

    # get indices of labeled frames
    dec_perc_labels = 0.01 * perc_labels
    n_labels = int(0.5 + dec_perc_labels * _y.data.shape[0])
    ind = non_zero_sum(_y)
    if len(ind) < n_labels:
        raise ValueError('Not enough labeled frames, label more frames or decrease label percentage.')

    # split labeled frames between train and val sets
    ind_train = np.random.choice(ind, size=n_labels, replace=False).tolist()
    ind_val = list_diff(ind, ind_train)
    assert len(ind_train) + len(ind_val) == len(ind)

    # create train and val sets
    x_train, y_train = create_train_set(_x, _y, ind_val, new_axes)
    x_val, y_val = create_val_set(_x, _y, ind_val)  # val sets without one-hot encoding

    # add channel dim and one-hot encoding
    X = x_train[..., np.newaxis]
    Y = convert_to_oneHot(y_train)
    X_val = x_val[..., np.newaxis]
    Y_val = convert_to_oneHot(y_val)

    return X, Y, X_val, Y_val, y_val, new_axes


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


# TODO: how do warnings show up in napari?
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


def train(model, training_data, validation_X, validation_Y, epochs, steps_per_epoch):
    # start training
    model.keras_model.fit(training_data, validation_data=(validation_X, validation_Y),
                          epochs=epochs, steps_per_epoch=steps_per_epoch,
                          callbacks=model.callbacks, verbose=1)

    # save last weights
    if model.basedir is not None:
        model.keras_model.save_weights(str(model.logdir / 'weights_last.h5'))
