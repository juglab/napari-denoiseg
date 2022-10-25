import warnings
from pathlib import Path
from queue import Queue
from typing import Union

import numpy as np

from tensorflow.keras.callbacks import Callback

from napari.qt.threading import thread_worker
import napari.utils.notifications as ntf

from denoiseg.utils.seg_utils import convert_to_oneHot
from denoiseg.utils.denoiseg_data_preprocessing import augment_patches
from tensorflow.python.framework.errors_impl import ResourceExhaustedError, UnknownError, NotFoundError

from napari_denoiseg.utils import (
    State,
    UpdateType,
    list_diff,
    reshape_data,
    optimize_threshold,
    load_model
)
from napari_denoiseg.utils import cwd, get_default_path


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

    def on_train_crashed(self):
        self.queue.put(UpdateType.CRASHED)

    def stop_training(self):
        self.model.stop_training = True


@thread_worker(start_thread=False)
def training_worker(widget, pretrained_model=None, expert_settings=None):
    import os
    import threading
    from napari_denoiseg.utils import UpdateType, generate_config

    ntf.show_info('Loading data')

    # patch shape
    patch_shape_XY = widget.patch_size_XY.value()
    patch_shape_Z = widget.patch_size_Z.value()
    if widget.is_3D:
        patch_shape = (patch_shape_Z, patch_shape_XY, patch_shape_XY)
    else:
        patch_shape = (patch_shape_XY, patch_shape_XY)

    ntf.show_info('Creating patches')
    # get images and labels
    # TODO we should make this list compatible, with the patch creation already
    X_train, Y_train, X_val, Y_val_onehot, Y_val, widget.new_axes = load_images(widget, patch_shape)

    # create DenoiSeg configuration
    n_epochs = widget.n_epochs
    n_steps = widget.n_steps
    batch_size = widget.batch_size_spin.value()

    ntf.show_info('Creating configuration')
    # create configuration
    if expert_settings is None:
        denoiseg_conf = generate_config(X_train, patch_shape, n_epochs, n_steps, batch_size)
    else:
        denoiseg_conf = generate_config(X_train,
                                        patch_shape,
                                        n_epochs,
                                        n_steps,
                                        batch_size,
                                        **expert_settings.get_settings())

    ntf.show_info('Preparing training')

    # load model if requested
    if expert_settings is not None:
        # priority is given to pretrained model, even if the expert settings point to one (must be a file)
        # i.e. priority to the most recently trained
        if expert_settings.has_model() and pretrained_model is None:
            # TODO check if models are compatible
            pretrained_model = load_model(expert_settings.get_model_path())

    # prepare training
    args = (widget, denoiseg_conf, X_train, Y_train, X_val, Y_val_onehot, pretrained_model)
    train_args, widget.tf_version = prepare_training(*args)
    denoiseg_updater = train_args[-1]

    # start training
    ntf.show_info('Training')
    training = threading.Thread(target=train, args=train_args)
    training.start()

    # loop looking for update events using the updater queue
    while True:
        update = denoiseg_updater.queue.get(True)

        if update == UpdateType.DONE or update == UpdateType.CRASHED:
            break
        elif widget.state != State.RUNNING:
            denoiseg_updater.stop_training()
            break
        else:
            yield update

    yield {UpdateType.TRAINING_DONE: ''}

    # training done, keep model in memory
    widget.model = train_args[0]

    # threshold validation data to estimate the best threshold
    ntf.show_info('Optimizing threshold')
    yield from get_best_threshold(widget, X_val, Y_val_onehot)

    # save input/output for bioimage.io
    with cwd(get_default_path()):
        widget.inputs = os.path.join(widget.model.basedir, 'inputs.npy')
        widget.outputs = os.path.join(widget.model.basedir, 'outputs.npy')
        np.save(widget.inputs, X_val[..., 0][np.newaxis, 0, ..., np.newaxis])
        np.save(widget.outputs, widget.model.predict(X_val[..., 0][np.newaxis, 0, ..., np.newaxis],
                                                     axes=widget.new_axes))

    ntf.show_info('Done')


def get_best_threshold(widget, X_val, Y_val):
    gen = optimize_threshold(widget.model, X_val, Y_val, widget.new_axes, widget=widget)
    best_threshold = -1
    best_score = -1
    while True:
        t = next(gen, None)

        if t:
            i_t, temp_threshold, temp_score = t

            if temp_score > best_score:
                best_score = temp_score
                best_threshold = temp_threshold

            yield {UpdateType.THRESHOLD: (i_t, temp_threshold)}
        else:
            break

    yield {UpdateType.BEST_THRESHOLD: (best_threshold, best_score)}


def load_images(widget, patch_shape=(256, 256)):
    """
    Load images from the disk or from napari layers.
    :param widget:
    :param patch_shape:
    :return:
    """
    # TODO make clearer what objects are returned

    # get axes
    axes = widget.axes_widget.get_axes()

    # get images and labels
    if widget.load_from_disk:  # from local folders
        path_train_X = widget.train_images_folder.get_folder()
        path_train_Y = widget.train_labels_folder.get_folder()
        path_val_X = widget.val_images_folder.get_folder()
        path_val_Y = widget.val_labels_folder.get_folder()

        if not Path(path_train_X).exists():
            ntf.show_error('X train folder doesn\'t exist')

        if not Path(path_train_Y).exists():
            ntf.show_error('Y train folder doesn\'t exist')

        if not Path(path_val_X).exists():
            ntf.show_error('X val folder doesn\'t exist')

        if not Path(path_val_Y).exists():
            ntf.show_error('Y val folder doesn\'t exist')

        return prepare_data_disk(path_train_X, path_train_Y, path_val_X, path_val_Y, axes, patch_shape)

    else:  # from layers
        image_data = widget.images.value.data
        label_data = widget.labels.value.data

        # split train and val
        perc_labels = widget.perc_train_slider.slider.get_value()

        return prepare_data_layers(image_data, label_data, perc_labels, axes)


def load_data_from_disk(source: Union[str, Path],
                        target: Union[str, Path],
                        axes: str,
                        augmentation=False,
                        check_exists=True,
                        patch_shape=(256, 256)):
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
    :param target: Path to the folder containing the label images.
    :param axes: Axes order
    :param augmentation: Augment data 8-fold if True
    :param check_exists: Check if source images have a target if True. Else it will load an empty image instead.
    :param patch_shape: Patch shape used if we are creating patches directly from data from the disk.
    :return:
    """
    from denoiseg.utils.seg_utils import convert_to_oneHot
    from denoiseg.utils.denoiseg_data_preprocessing import generate_patches_from_list
    from napari_denoiseg.utils import load_pairs_from_disk

    # load train data
    _x, _y, _axes = load_pairs_from_disk(source, target, axes, check_exists=check_exists)

    # if lists, then generate patches already
    if type(_x) == list:
        new_axes = _axes

        # first we need to reshape each array
        reshaped_x, reshaped_y = [], []
        for im_x, im_y in zip(_x, _y):
            x_r, y_r, new_axes = reshape_data(im_x, im_y, _axes)
            reshaped_x.append(x_r)
            reshaped_y.append(y_r)

        x_np, y_np = generate_patches_from_list(reshaped_x, reshaped_y, axes=new_axes, shape=patch_shape)

        # one-hot encoding
        Y = convert_to_oneHot(y_np)
    else:
        # reshape data
        x_np, y_np, new_axes = reshape_data(_x, _y, _axes)

        # apply augmentation, XY dimensions must be equal (dim(X) == dim(Y))
        if augmentation:
            x_np = augment_patches(x_np, new_axes)
            y_np = augment_patches(y_np, new_axes)
            print('Raw image size after augmentation', x_np.shape)
            print('Mask size after augmentation', y_np.shape)

        # add channel dim and one-hot encoding
        Y = convert_to_oneHot(y_np)

    return x_np, Y, y_np, new_axes


def prepare_data_disk(path_train_X, path_train_Y, path_val_X, path_val_Y, axes, patch_shape=(256, 256)):
    #
    (X, Y, _, new_axes) = load_data_from_disk(path_train_X,
                                              path_train_Y,
                                              axes,
                                              augmentation=True,
                                              check_exists=False,
                                              patch_shape=patch_shape)
    (X_val, Y_val, y_val, _) = load_data_from_disk(path_val_X, path_val_Y, axes)

    return X, Y, X_val, Y_val, y_val, new_axes


def detect_non_zero_frames(im):
    """
    Detect empty slices along the 0th dim.

    :param im: Image stack
    :return:
    """
    assert len(im.shape) > 2

    if im.shape[0] > 1:
        im_reshaped = im.reshape(im.shape[0], -1)

        return np.where(np.sum(im_reshaped != 0, axis=1) != 0)[0]
    else:
        if im.min() == im.max() == 0:
            return [0]
        else:
            return []


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

    # Y does not have channels dim before one hot encoding
    masks = np.zeros(x.shape[:-1])

    # missing frames are replaced by empty ones
    masks[:y.shape[0], ...] = y

    x_aug = augment_patches(np.delete(x, ind_exclude, axis=0), axes)
    y_aug = augment_patches(np.delete(masks, ind_exclude, axis=0), axes)

    return x_aug, y_aug


def create_val_set(x, y, ind_include):
    """

    :param x:
    :param y:
    :param ind_include:
    :return:
    """
    return np.take(x, ind_include, axis=0), np.take(y, ind_include, axis=0)


def check_napari_data(x, y, axes: str):
    """

    :param x:
    :param y:
    :param axes:
    :return:
    """

    if axes[-2:] != 'YX':
        raise ValueError('X and Y axes are in the wrong order.')

    if len(x.shape) < 3:
        raise ValueError('Images must have a 3rd dimension (multiple samples).')

    if len(axes) != len(x.shape):
        raise ValueError('Raw images dimensions and axes are incompatible.')

    if 'C' in axes:
        if len(axes) != len(y.shape) + 1:
            raise ValueError('Label images dimensions and axes are incompatible.')

        if len(x.shape) != len(y.shape) + 1:
            raise ValueError('Raw and label images dimensions are incompatible.')
    else:
        if len(axes) != len(y.shape):
            raise ValueError('Label images dimensions and axes are incompatible.')

        if len(x.shape) != len(y.shape):
            raise ValueError('Raw and label images dimensions are incompatible.')

    # X and Y dims are fixed in napari, check that they are equal for the augmentation
    if x.shape[-1] != x.shape[-2]:
        raise ValueError('Raw data X and Y dimensions should be equal.')

    if y.shape[-1] != y.shape[-2]:
        raise ValueError('Label image X and Y dimensions should be equal.')

    if x.shape[-1] != y.shape[-1] or x.shape[-2] != y.shape[-2]:
        raise ValueError('Raw and labels have different X and Y dimensions.')


def prepare_data_layers(raw, gt, perc_labels, axes, previous_label_indices=None):
    """

    perc_labels: ]0-100[

    :param raw:
    :param gt:
    :param perc_labels:
    :param axes
    :param previous_label_indices:
    :return:
    """

    # sanity check on the data
    check_napari_data(raw, gt, axes)

    # reshape data
    _x, _y, new_axes = reshape_data(raw, gt, axes)

    # get indices of labeled frames
    label_indices = detect_non_zero_frames(_y)

    # get number of requested training labels
    # TODO fix the problems in case of continue training
    # check length of given indices vs perc result. If different then ignore perc
    dec_perc_labels = 0.01 * perc_labels
    n_labels = len(label_indices)
    n_train_labels = int(0.5 + dec_perc_labels * n_labels)

    if perc_labels == 0 or perc_labels == 100:
        err = 'Percentage of training labels cannot be 0 or 100%.'
        ntf.show_error(err)
        raise ValueError(err)
    if len(label_indices) == 0 or n_train_labels < 5:
        err = 'Not enough labeled images for training, label more frames or decrease label percentage.'
        ntf.show_error(err)
        raise ValueError(err)
    if n_labels - n_train_labels < 2:
        err = 'Not enough labeled images for validation, label more frames or decrease label percentage.'
        ntf.show_error(err)
        raise ValueError(err)

    # split labeled frames between train and val sets
    # we do not split randomly in case we continue training
    # TODO fix the problems in case of continue training
    ind_train = np.random.choice(label_indices, size=n_train_labels, replace=False).tolist()
    ind_val = list_diff(label_indices, ind_train)
    assert len(ind_train) + len(ind_val) == len(label_indices)

    # create train and val sets
    X, y_train = create_train_set(_x, _y, ind_val, new_axes)
    X_val, y_val_no_hot = create_val_set(_x, _y, ind_val)  # val sets without one-hot encoding

    # add channel dim and one-hot encoding
    Y = convert_to_oneHot(y_train)
    Y_val = convert_to_oneHot(y_val_no_hot)

    return X, Y, X_val, Y_val, y_val_no_hot, new_axes


def prepare_training(widget, conf, X_train, Y_train, X_val, Y_val, pretrained_model=None):
    import tensorflow as tf
    from denoiseg.models import DenoiSeg
    from csbdeep.utils import axes_check_and_normalize
    from denoiseg.internals.DenoiSeg_DataWrapper import DenoiSeg_DataWrapper
    from n2v.utils import n2v_utils
    from n2v.utils.n2v_utils import pm_uniform_withCP

    # create model
    # model_name = today + '_DenoiSeg_' + str(round(time.time()))
    with cwd(get_default_path()):
        model_name = 'DenoiSeg_3D' if widget.is_3D else 'DenoiSeg_2D'
        base_dir = Path('models').absolute()

        model = DenoiSeg(conf, model_name, base_dir)

    # if tf.config.list_physical_devices('GPU'):
    #    tf.config.experimental.set_memory_growth(tf.config.list_physical_devices('GPU')[0], True)

    # if we continue training, transfer the weights
    # TODO: currently not reusing the same model in case the configuration changed (n epochs, data etc.)
    if pretrained_model:
        # TODO use the configurations to check whether the networks are compatible
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
    # Patches are created here
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
    # TODO validation data loader
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
    train_params = (model, training_data, validation_X, validation_Y, epochs, steps_per_epoch, updater)

    return train_params, tf.__version__


def sanity_check_validation_fraction(X_train, X_val):
    import warnings
    n_train, n_val = len(X_train), len(X_val)
    frac_val = (1.0 * n_val) / (n_train + n_val)
    frac_warn = 0.05
    if frac_val < frac_warn:
        message = "small number of validation images (only %.05f%% of all images)" % (100 * frac_val)
        warnings.warn(message)
        ntf.show_info(message)


def sanity_check_training_size(X_train, model, axes):
    from csbdeep.utils import axes_dict

    ax = axes_dict(axes)
    axes_relevant = ''.join(a for a in 'XYZT' if a in axes)
    div_by = 2 ** model.config.unet_n_depth
    for a in axes_relevant:
        n = X_train.shape[ax[a]]
        if n % div_by != 0:
            err = "training images must be evenly divisible by %d along axes %s (axis %s has " \
                  "incompatible size %d)" % (div_by, axes_relevant, a, n)
            ntf.show_error(err)
            raise ValueError(err)


def get_validation_patch_shape(X_val, axes):
    from csbdeep.utils import axes_dict

    ax = axes_dict(axes)
    axes_relevant = ''.join(a for a in 'XYZT' if a in axes)  # TODO: why T in there? It shouldn't be in T
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


def train_error(updater, args, msg: str):
    # TODO: napari 0.4.16 has ntf.show_error, but napari workflows requires 0.4.15 that doesn't
    # ntf.show_error(msg)
    ntf.show_info(msg)
    warnings.warn(msg)
    print(args)
    updater.on_train_crashed()


def train(model, training_data, validation_X, validation_Y, epochs, steps_per_epoch, updater):
    with cwd(get_default_path()):
        try:
            # start training
            model.keras_model.fit(training_data,
                                  validation_data=(validation_X, validation_Y),
                                  epochs=epochs,
                                  steps_per_epoch=steps_per_epoch,
                                  callbacks=model.callbacks,
                                  verbose=1)
            # save last weights
            if model.basedir is not None:
                model.keras_model.save_weights(str(model.logdir / 'weights_last.h5'))

        except MemoryError as e:
            msg = 'MemoryError can be an OOM error on the GPU (reduce batch and/or patch size, close other processes).'
            train_error(updater, str(e), msg)
        except ResourceExhaustedError as e:
            msg = 'ResourceExhaustedError can be an OOM error on the GPU (reduce batch and/or patch size)'
            train_error(updater, e.message, msg)
        except (NotFoundError, UnknownError) as e:
            msg = 'NotFoundError or UnknownError can be caused by an improper loading of cudnn, try restarting.'
            train_error(updater, e.message, msg)

