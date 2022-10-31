import os
from pathlib import Path
import warnings
from contextlib import contextmanager
from enum import Enum
from typing import Union

import numpy as np
from itertools import permutations


from napari_denoiseg.resources import DOC_BIOIMAGE

REF_AXES = 'TSZYXC'

# this is arbitrary, but allows having S same for input to the plugin and output
# eg. input is SYX, DenoiSeg output are SYX (denoised) and CSYX (seg). If C is placed after S, then we cannot compare
# input and output on the same axes anymore.
NAPARI_AXES = 'CTSZYX'


class State(Enum):
    IDLE = 0
    RUNNING = 1
    INTERRUPTED = 2


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
    TRAINING_DONE = 'training done'
    CRASHED = 'crashed'
    DONE = 'done'
    RETRAIN = 'retrain'


def build_modelzoo(path: Union[str, Path], weights: str, inputs, outputs, tf_version: str, axes='byxc'):
    import os
    from bioimageio.core.build_spec import build_model

    assert path.endswith('.bioimage.io.zip'), 'Path must end with .bioimage.io.zip'

    tags_dim = '3d' if len(axes) == 5 else '2d'
    doc = DOC_BIOIMAGE

    head, _ = os.path.split(str(weights))
    head = os.path.join(os.path.normcase(head), "config.json")
    build_model(weight_uri=str(weights),
                test_inputs=[inputs],
                test_outputs=[outputs],
                input_axes=[axes],
                output_axes=[axes],
                output_path=path,
                name='DenoiSeg',
                description="Super awesome DenoiSeg model. The best.",
                authors=[{"name": "Tim-Oliver Buchholz"},
                         {"name": "Mangal Prakash"},
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
                tensorflow_version=tf_version,
                attachments={"files": head}
                )


def get_shape_order(shape_in, axes_in, ref_axes):
    """
    Return the new shape and axes order of x, if the axes were to be ordered according to
    the reference axes.

    :param shape_in:
    :param ref_axes: Reference axes order (string)
    :param axes_in: New axes as a list of strings
    :return:
    """
    assert len(shape_in) == len(axes_in), 'Mismatch between shape and axes sizes'

    # build indices look-up table: indices of each axis in `axes`
    indices = [axes_in.find(k) for k in ref_axes]

    # remove all non-existing axes (index == -1)
    indices = list(filter(lambda k: k != -1, indices))

    # find axes order and get new shape
    new_axes = [axes_in[ind] for ind in indices]
    new_shape = tuple([shape_in[ind] for ind in indices])

    return new_shape, ''.join(new_axes), indices


def list_diff(l1, l2):
    """
    Return the difference of two lists.
    :param l1:
    :param l2:
    :return: list of elements in l1 that are not in l2.
    """
    return list(set(l1) - set(l2))


def remove_C_dim(shape, axes):
    ind = axes.find('C')

    if ind == -1:
        return shape

    return *shape[:ind], *shape[ind + 1:]


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
    new_x_shape, new_axes, indices = get_shape_order(_x.shape, _axes, REF_AXES)

    if 'C' in _axes:  # Y does not have a C dimension
        index_c = indices[-1]
        indices_y = indices[:-1]  # remove position of C

        # correct the axes that were places after C
        for i, ind in enumerate(indices_y):
            if ind > index_c:
                indices_y[i] = ind - 1

        axes_y = _axes.replace('C', '')
        ref_axes_y = REF_AXES.replace('C', '')
        new_y_shape, _, _ = get_shape_order(_y.shape, axes_y, ref_axes_y)

    else:
        new_y_shape = tuple([_y.shape[ind] for ind in indices])
        indices_y = indices

    # if S is not in the list of axes, then add a singleton S
    if 'S' not in new_axes:
        new_axes = 'S' + new_axes
        _x = _x[np.newaxis, ...]
        _y = _y[np.newaxis, ...]
        new_x_shape = (1,) + new_x_shape
        new_y_shape = (1,) + new_y_shape

        # need to change the array of indices
        indices = [0] + [1 + i for i in indices]
        indices_y = [0] + [1 + i for i in indices_y]

    # reshape by moving axes
    destination = [i for i in range(len(indices))]
    destination_y = [i for i in range(len(indices_y))]
    _x = np.moveaxis(_x, indices, destination)
    _y = np.moveaxis(_y, indices_y, destination_y)

    # remove T if necessary
    if 'T' in new_axes:
        new_x_shape = (-1,) + new_x_shape[2:]  # remove T and S
        new_y_shape = (-1,) + new_y_shape[2:]
        new_axes = new_axes.replace('T', '')

        # reshape arrays
        _x = _x.reshape(new_x_shape)
        _y = _y.reshape(new_y_shape)

    # add channel
    if 'C' not in new_axes:
        _x = _x[..., np.newaxis]
        new_axes = new_axes + 'C'

    return _x, _y, new_axes


def reshape_data_single(x, axes: str):
    """
    Same as reshape_data but for a single array
    """
    _x = x
    _axes = axes

    # sanity checks
    if 'X' not in axes or 'Y' not in axes:
        raise ValueError('X or Y dimension missing in axes.')

    if len(_axes) != len(_x.shape):
        raise ValueError('Incompatible data and axes.')

    assert len(list_diff(list(_axes), list(REF_AXES))) == 0, 'Unknown axes'  # all axes are part of REF_AXES

    # get new x shape
    new_x_shape, new_axes, indices = get_shape_order(_x.shape, _axes, REF_AXES)

    # if S is not in the list of axes, then add a singleton S
    if 'S' not in new_axes:
        new_axes = 'S' + new_axes
        _x = _x[np.newaxis, ...]
        new_x_shape = (1,) + new_x_shape

        # need to change the array of indices
        indices = [0] + [1 + i for i in indices]

    # reshape by moving axes
    destination = [i for i in range(len(indices))]
    _x = np.moveaxis(_x, indices, destination)

    # remove T if necessary
    if 'T' in new_axes:
        new_x_shape = (-1,) + new_x_shape[2:]  # remove T and S
        new_axes = new_axes.replace('T', '')

        # reshape S and T together
        _x = _x.reshape(new_x_shape)

    # add channel
    if 'C' not in new_axes:
        _x = _x[..., np.newaxis]
        new_axes = new_axes + 'C'

    return _x, new_axes


# TODO rename to reshape arbitrary?
def reshape_napari(x, axes_in: str, axes_out: str = NAPARI_AXES):
    """
    Reshape the data according to the napari axes order (or any order if axes_out is set).
    """
    _x = x
    _axes = axes_in

    # sanity checks
    if 'X' not in axes_in or 'Y' not in axes_in:
        raise ValueError('X or Y dimension missing in axes.')

    if len(_axes) != len(_x.shape):
        raise ValueError('Incompatible data and axes.')

    assert len(list_diff(list(_axes), list(REF_AXES))) == 0  # all axes are part of REF_AXES

    # get new x shape
    new_x_shape, new_axes, indices = get_shape_order(_x.shape, _axes, axes_out)

    # reshape by moving the axes
    destination = [i for i in range(len(indices))]
    _x = np.moveaxis(_x, indices, destination)

    return _x, new_axes


# TODO write tests
def get_napari_shapes(shape_in, axes_in):
    """
    Transform shape into what DenoiSeg expect and return the denoised and segmented output shapes in napari axes order.

    :param shape_in:
    :param axes_in:
    :return:
    """
    # TODO where is this called? does it suffer from the same issues than reshape ?
    # shape and axes for DenoiSeg
    shape_denoiseg, denoiseg_axes, _ = get_shape_order(shape_in, axes_in, REF_AXES)

    # denoised and segmented image shapes
    if 'C' in axes_in:
        shape_denoised = shape_denoiseg
        shape_segmented = (*shape_denoiseg[:-1], 3)
        segmented_axes = denoiseg_axes
    else:
        shape_denoised = shape_denoiseg
        shape_segmented = (*shape_denoiseg, 3)
        segmented_axes = denoiseg_axes + 'C'

    # shape and axes for napari
    shape_denoised_out, _, _ = get_shape_order(shape_denoised, denoiseg_axes, NAPARI_AXES)
    shape_segmented_out, _, _ = get_shape_order(shape_segmented, segmented_axes, NAPARI_AXES)

    return shape_denoised_out, shape_segmented_out


def get_default_path():
    return os.path.join(Path.home(), ".napari", "DenoiSeg")


@contextmanager
def cwd(path):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)
    oldpwd = os.getcwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(oldpwd)
