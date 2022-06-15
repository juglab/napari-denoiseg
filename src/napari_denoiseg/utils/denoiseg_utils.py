from pathlib import Path

from enum import Enum
import numpy as np
from tifffile import imread

from csbdeep.data import RawData
from csbdeep.utils import consume, axes_check_and_normalize
from denoiseg.models import DenoiSeg


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
def from_folder(source_dir, target_dir, axes='CZYX', check_exists=True):
    """
    Builds a generator for pairs of source and target images with same names. `check_exists` = `False` allows inserting
    empty images when the corresponding target is not found.

    Adapted from RawData.from_folder in CSBDeep.

    :param source_dir: Absolute path to folder containing source images
    :param target_dir: Absolute path to folder containing target images, with same names than in `source_folder`
    :param axes: Semantics of axes of loaded images (assumed to be the same for all images).
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

    # sanity check on the axes
    axes = axes_check_and_normalize(axes)

    # generate description
    n_images = len(pairs)
    description = "{p}: target='{o}', sources={s}, axes='{a}', pattern='{pt}'".format(p=s.parent,
                                                                                      s=s.name,
                                                                                      o=t.name, a=axes,
                                                                                      pt=pattern)

    def _gen():
        for fx, fy in pairs:
            if fy:  # read images
                x, y = imread(str(fx)), imread(str(fy))
            else:  # if the target is None, replace by an empty image
                x = imread(str(fx))
                y = np.zeros(x.shape)

            len(axes) >= x.ndim or _raise(ValueError())
            yield x, y, axes[-x.ndim:], None

    return RawData(_gen, n_images, description)


def generate_config(X, n_epochs=20, n_steps=400, batch_size=16, patch_size=64):
    from denoiseg.models import DenoiSegConfig

    patch_shape = tuple([int(x) for x in np.repeat(patch_size, len(X.shape) - 2)])  # TODO: won't work for 3D
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
    if weights_path[-4:] == ".zip":
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
def load_from_disk(path):
    """

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

    return np.array(images) if dims_agree else images


def load_pairs_from_disk(source_path, target_path, axes='CYX', check_exists=True):
    """

    :param source_path:
    :param target_path:
    :param axes:
    :param check_exists:
    :return:
    """
    # create RawData generator
    pairs = from_folder(source_path, target_path, axes, check_exists)

    # load data
    _source = []
    _target = []
    for s, t, _, _ in pairs.generator():
        _source.append(s)
        _target.append(t)

    return np.array(_source), np.array(_target, dtype=np.int)


def build_modelzoo(path, weights, inputs, outputs, tf_version, doc='../resources/documentation.md'):
    import os
    from bioimageio.core.build_spec import build_model

    assert path.endswith('.bioimage.io.zip'), 'Path must end with .bioimage.io.zip'
    build_model(weight_uri=weights,
                test_inputs=[inputs],
                test_outputs=[outputs],
                input_axes=["byxc"],
                output_axes=["byxc"],
                output_path=path,
                name='DenoiSeg',
                description="Super awesome DenoiSeg model. The best.",
                authors=[{"name": "Tim-Oliver Buchholz"}, {"name": "Mangal Prakash"},
                         {"name": "Alexander Krull"},
                         {"name": "Florian Jug"}],
                license="BSD-3-Clause",
                documentation=os.path.abspath(doc),
                tags=["2d", "tensorflow", "unet", "denoising", "semantic-segmentation"],
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
