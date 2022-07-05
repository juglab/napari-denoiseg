import os
import warnings
from pathlib import Path

from enum import Enum
import numpy as np
from denoiseg.utils.compute_precision_threshold import measure_precision
from tifffile import imread

from csbdeep.data import RawData
from csbdeep.utils import consume
from denoiseg.models import DenoiSeg
from itertools import permutations

REF_AXES = 'TSZYXC'


class State(Enum):
    IDLE = 0
    RUNNING = 1


class ModelSaveMode(Enum):
    MODELZOO = 'Bioimage.io'
    TF = 'TensorFlow'

    @classmethod
    def list(cls):
        return list(map(lambda c: c.value, cls))


class UpdateType(Enum):
    EPOCH = 'epoch'
    BATCH = 'batch'
    LOSS = 'loss'
    N_IMAGES = 'number of images'
    IMAGE = 'image'
    DONE = 'done'


# Adapted from:
# https://csbdeep.bioimagecomputing.com/doc/_modules/csbdeep/data/rawdata.html#RawData.from_folder
def load_pairs_generator(source_dir, target_dir, axes, check_exists=True):
    """
    Builds a generator for pairs of source and target images with same names. `check_exists` = `False` allows inserting
    empty images when the corresponding target is not found.

    Adapted from RawData.from_folder in CSBDeep.

    :param axes:
    :param source_dir: Absolute path to folder containing source images
    :param target_dir: Absolute path to folder containing target images, with same names than in `source_folder`
    :param check_exists: If `True`, raises an exception if a target is missing, target is set to `None` if `check_exist`
                        is `False`.
    :return:`RawData` object, whose `generator` is used to yield all matching TIFF pairs.
            The generator will return a tuple `(x,y,axes,mask)`, where `x` is from
            `source_dirs` and `y` is the corresponding image from the `target_dir`;
            `mask` is set to `None`.
    """

    def substitute_by_none(tuple_list, ind):
        """
        Substitute the second element in tuple `ind` with `None`
        :param tuple_list: List of tuples
        :param ind: Index of the tuple in which to substitute the second element with `None`
        :return:
        """
        tuple_list[ind] = (tuple_list[ind][0], None)

    def _raise(e):
        raise e

    # pattern of images to select
    pattern = '*.tif*'

    # list of possible pairs based on the file found in the source folder
    s = Path(source_dir)
    t = Path(target_dir)
    pairs = [(f, t / f.name) for f in s.glob(pattern)]
    if len(pairs) == 0:
        raise FileNotFoundError("Didn't find any images.")

    # check if the corresponding target exists
    if check_exists:
        consume(t.exists() or _raise(FileNotFoundError(t)) for s, t in pairs)
    else:
        # alternatively, replace non-existing files with None
        consume(p[1].exists() or substitute_by_none(pairs, i) for i, p in enumerate(pairs))

    # generate description
    n_images = len(pairs)
    description = "{p}: target='{o}', sources={s}, axes='{a}', pattern='{pt}'".format(p=s.parent,
                                                                                      s=s.name,
                                                                                      o=t.name, a=axes,
                                                                                      pt=pattern)

    # keep C index in memory
    c_in_axes = 'C' in axes
    index_c = axes.find('C')

    def _gen():
        for fx, fy in pairs:
            if fy:  # read images
                x, y = imread(str(fx)), imread(str(fy))
            else:  # if the target is None, replace by an empty image
                x = imread(str(fx))

                if c_in_axes:
                    new_shape = list(x.shape)
                    new_shape.pop(index_c)
                    y = np.zeros(new_shape)
                else:
                    y = np.zeros(x.shape)

            len(axes) >= x.ndim or _raise(ValueError())
            yield x, y

    return RawData(_gen, n_images, description)


def generate_config(X, patch_shape, n_epochs=20, n_steps=400, batch_size=16):
    from denoiseg.models import DenoiSegConfig

    # assert len(X.shape)-2 == len(patch_shape)
    # TODO: what if generator or list
    conf = DenoiSegConfig(X, unet_kern_size=3, n_channel_out=4, relative_weights=[1.0, 1.0, 5.0],
                          train_steps_per_epoch=n_steps, train_epochs=n_epochs,
                          batch_norm=True, train_batch_size=batch_size, n2v_patch_shape=patch_shape,
                          unet_n_first=32, unet_n_depth=4, denoiseg_alpha=0.5, train_tensorboard=True)

    return conf


def load_weights(model: DenoiSeg, weights_path):
    """

    :param model:
    :param weights_path:
    :return:
    """
    _filename, file_ext = os.path.splitext(weights_path)
    if file_ext == ".zip":
        import bioimageio.core
        # we assume we got a modelzoo file
        rdf = bioimageio.core.load_resource_description(weights_path)
        weights_name = rdf.weights['keras_hdf5'].source
    else:
        # we assure we have a path to a .h5
        weights_name = weights_path

    if not Path(weights_name).exists():
        raise FileNotFoundError('Invalid path to weights.')

    model.keras_model.load_weights(weights_name)


# TODO: we must make sure that if the function returns a list, then it is handled correctly by prediction
def load_from_disk(path, axes: str):
    """

    :param axes:
    :param path:
    :return:
    """
    images_path = Path(path)
    image_files = [f for f in images_path.glob('*.tif*')]

    images = []
    dims_agree = True
    for f in image_files:
        images.append(imread(str(f)))
        dims_agree = dims_agree and (images[0].shape == images[-1].shape)

    if dims_agree:
        if 'S' in axes:
            ind_S = axes.find('S')
            final_images = np.concatenate(images, axis=ind_S)
        else:
            final_images = np.stack(images, axis=0)
        return final_images

    return images


def lazy_load_generator(path):
    """

    :param path:
    :return:
    """
    images_path = Path(path)
    image_files = [f for f in images_path.glob('*.tif*')]

    def generator(file_list):
        counter = 0
        for f in file_list:
            counter = counter + 1
            yield imread(str(f)), f, counter

    return generator(image_files), len(image_files)


def load_pairs_from_disk(source_path, target_path, axes, check_exists=True):
    """

    :param axes:
    :param source_path:
    :param target_path:
    :param check_exists:
    :return:
    """
    # create RawData generator
    pairs = load_pairs_generator(source_path, target_path, axes, check_exists)
    n = pairs.size

    # load data
    _source = []
    _target = []
    for s, t in pairs.generator():
        _source.append(s)
        _target.append(t)

    _s = np.array(_source)
    _t = np.array(_target, dtype=np.int)

    if 'S' not in axes and n > 1:
        _axes = 'S' + axes
    else:
        _axes = axes

    if 'C' in axes:
        if remove_C_dim(_s.shape, _axes) != _t.shape:
            raise ValueError
    else:
        if _s.shape != _t.shape:
            raise ValueError

    return _s, _t, n


def build_modelzoo(path, weights, inputs, outputs, tf_version, axes='byxc', doc='../resources/documentation.md'):
    import os
    from bioimageio.core.build_spec import build_model

    assert path.endswith('.bioimage.io.zip'), 'Path must end with .bioimage.io.zip'

    tags_dim = '3d' if len(axes) == 5 else '2d'

    build_model(weight_uri=weights,
                test_inputs=[inputs],
                test_outputs=[outputs],
                input_axes=[axes],
                # TODO are the axes in and out always the same? (output has 3 seg classes and 1 denoised channels)
                output_axes=[axes],
                output_path=path,
                name='DenoiSeg',
                description="Super awesome DenoiSeg model. The best.",
                authors=[{"name": "Tim-Oliver Buchholz"}, {"name": "Mangal Prakash"},
                         {"name": "Alexander Krull"},
                         {"name": "Florian Jug"}],
                license="BSD-3-Clause",
                documentation=os.path.abspath(doc),
                tags=[tags_dim, "tensorflow", "unet", "denoising", "semantic-segmentation"],
                cite=[
                    {"text": "DenoiSeg: Joint Denoising and Segmentation", "doi": "10.48550/arXiv.2005.02987"}],
                preprocessing=[[{
                    "name": "zero_mean_unit_variance",
                    "kwargs": {
                        "axes": "yx",
                        "mode": "per_dataset"
                    }
                }]],
                tensorflow_version=tf_version
                )


def get_shape_order(x, ref_axes, axes):
    """
    Return the new shape and axes order of x, if the axes were to be ordered according to
    the reference axes.

    :param x:
    :param ref_axes: Reference axes order (string)
    :param axes: New axes as a list of strings
    :return:
    """
    # build indices look-up table: indices of each axe in `axes`
    indices = [axes.find(k) for k in ref_axes]

    # remove all non-existing axes (index == -1)
    indices = tuple(filter(lambda k: k != -1, indices))

    # find axes order and get new shape
    new_axes = [axes[ind] for ind in indices]
    new_shape = tuple([x.shape[ind] for ind in indices])

    return new_shape, ''.join(new_axes), indices


def list_diff(l1, l2):
    """
    Return the difference of two lists.
    :param l1:
    :param l2:
    :return: list of elements in l1 that are not in l2.
    """
    return list(set(l1) - set(l2))


def reshape_data(x, y, axes: str):
    """
    Reshape the data to 'SZXYC' depending on the available `axes`. If a T dimension is present, the different time
    points are considered independent and stacked along the S dimension.

    Differences between x and y:
    - y can have a different S and T dimension size
    - y doesn't have C dimension

    :param x: Raw data.
    :param y: Ground-truth data.
    :param axes: Current axes order of X
    :return: Reshaped x, reshaped y, new axes order
    """
    _x = x
    _y = y
    _axes = axes

    # sanity checks TODO: raise error rather than assert?
    if 'X' not in axes or 'Y' not in axes:
        raise ValueError('X or Y dimension missing in axes.')

    if 'C' in _axes:
        if not (len(_axes) == len(_x.shape) == len(_y.shape) + 1):
            raise ValueError('Incompatible data and axes.')
    else:
        if not (len(_axes) == len(_x.shape) == len(_y.shape)):
            raise ValueError('Incompatible data and axes.')

    assert len(list_diff(list(_axes), list(REF_AXES))) == 0  # all axes are part of REF_AXES

    # get new x shape
    new_x_shape, new_axes, indices = get_shape_order(_x, REF_AXES, _axes)

    if 'C' in _axes:  # Y does not have a C dimension
        axes_y = _axes.replace('C', '')
        ref_axes_y = REF_AXES.replace('C', '')
        new_y_shape, _, _ = get_shape_order(_y, ref_axes_y, axes_y)
    else:
        new_y_shape = tuple([_y.shape[ind] for ind in indices])

    # if S is not in the list of axes, then add a singleton S
    if 'S' not in new_axes:
        new_axes = 'S' + new_axes
        _x = _x[np.newaxis, ...]
        _y = _y[np.newaxis, ...]
        new_x_shape = (1,) + new_x_shape
        new_y_shape = (1,) + new_y_shape

    # remove T if necessary
    if 'T' in new_axes:
        new_x_shape = (-1,) + new_x_shape[2:]  # remove T and S
        new_y_shape = (-1,) + new_y_shape[2:]
        new_axes = new_axes.replace('T', '')

    # reshape
    _x = _x.reshape(new_x_shape)
    _y = _y.reshape(new_y_shape)

    # add channel
    if 'C' not in new_axes:
        _x = _x[..., np.newaxis]
        new_axes = new_axes + 'C'

    return _x, _y, new_axes


# TODO: this is a copy of reshape_data but without Y...
def reshape_data_single(x, axes: str):
    """
    """
    _x = x
    _axes = axes

    # sanity checks
    if 'X' not in axes or 'Y' not in axes:
        raise ValueError('X or Y dimension missing in axes.')

    if len(_axes) != len(_x.shape):
        raise ValueError('Incompatible data and axes.')

    assert len(list_diff(list(_axes), list(REF_AXES))) == 0  # all axes are part of REF_AXES

    # get new x shape
    new_x_shape, new_axes, indices = get_shape_order(_x, REF_AXES, _axes)

    # if S is not in the list of axes, then add a singleton S
    if 'S' not in new_axes:
        new_axes = 'S' + new_axes
        _x = _x[np.newaxis, ...]
        new_x_shape = (1,) + new_x_shape

    # remove T if necessary
    if 'T' in new_axes:
        new_x_shape = (-1,) + new_x_shape[2:]  # remove T and S
        new_axes = new_axes.replace('T', '')

    # reshape
    _x = _x.reshape(new_x_shape)

    # add channel
    if 'C' not in new_axes:
        _x = _x[..., np.newaxis]
        new_axes = new_axes + 'C'

    return _x, new_axes


def remove_C_dim(shape, axes):
    ind = axes.find('C')

    if ind == -1:
        return shape

    return (*shape[:ind], *shape[ind + 1:])


def filter_dimensions(shape_length, is_3D):
    """
    """
    axes = list(REF_AXES)
    axes.remove('Y')  # skip YX, constraint
    axes.remove('X')
    n = shape_length - 2

    if not is_3D:  # if not 3D, remove it from the
        axes.remove('Z')

    if n > len(axes):
        warnings.warn('Data shape length is too large.')
        return []
    else:
        all_permutations = [''.join(p) + 'YX' for p in permutations(axes, n)]

        if is_3D:
            all_permutations = [p for p in all_permutations if 'Z' in p]

        if len(all_permutations) == 0 and not is_3D:
            all_permutations = ['YX']

        return all_permutations


def are_axes_valid(axes: str):
    _axes = axes.upper()

    # length 0 and >6 are not accepted
    if 0 > len(_axes) > 6:
        return False

    # all characters must be in REF_AXES = 'STZYXC'
    if not all([s in REF_AXES for s in _axes]):
        return False

    # check for repeating characters
    for i, s in enumerate(_axes):
        if i != _axes.rfind(s):
            return False

    return True


def optimize_threshold(model, image_data, label_data, axes, widget=None, measure=measure_precision):
    """

    :return:
    """
    for i, ts in enumerate(np.linspace(0.1, 1, 19)):
        _, _, score, _ = model.predict_label_masks(image_data, label_data, axes[1:], ts, measure())

        # check if stop requested
        if widget is not None and widget.state != State.RUNNING:
            break

        yield i, ts, score


def reshape_napari(x, axes: str):
    """

    """
    _x = x
    _axes = axes

    # sanity checks
    if 'X' not in axes or 'Y' not in axes:
        raise ValueError('X or Y dimension missing in axes.')

    if len(_axes) != len(_x.shape):
        raise ValueError('Incompatible data and axes.')

    assert len(list_diff(list(_axes), list(REF_AXES))) == 0  # all axes are part of REF_AXES

    # get new x shape
    new_x_shape, new_axes, indices = get_shape_order(_x, 'SCTZYX', _axes)

    # reshape
    _x = _x.reshape(new_x_shape)

    return _x, new_axes

