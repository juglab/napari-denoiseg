from pathlib import Path
from typing import Union

import numpy as np

from tifffile import imread

from napari.utils import notifications as ntf

from csbdeep.data import RawData
from csbdeep.utils import consume

from .denoiseg_utils import remove_C_dim


def load_pairs_from_disk(source_path: Union[str, Path], target_path: Union[str, Path], axes: str, check_exists=True):
    """
    Load pairs of source and target images from the disk. This method returns arrays if all source images have the
    same shape or lists of images if they do not.
    `check_exists` = `False` allows inserting empty images when the corresponding target is not found.
    :param axes: Name of the expected axes of the images
    :param source_path: Path to the source images
    :param target_path: Path to the target images
    :param check_exists: Replace missing targets by empty images if `False`, ignore source images without targets
    otherwise.
    :return:
    """
    # create RawData generator
    pairs = load_pairs_generator(source_path, target_path, axes, check_exists)

    # load data
    _source = []
    _target = []
    dims_agree = True
    for s, t in pairs.generator():
        _source.append(s)
        _target.append(t)

        # check if the dimension agrees with the first image in the array
        dims_agree = dims_agree and (_source[0].shape == _source[-1].shape)

    # if all source images have the same shape
    if len(_source) > 0 and dims_agree:
        # concatenate in a single array
        if 'S' in axes:
            ind_S = axes.find('S')
            _s = np.concatenate(_source, axis=ind_S)
            _t = np.concatenate(_target, axis=ind_S)
            _axes = axes
        else:
            _axes = 'S' + axes
            _s = np.stack(_source, axis=0)
            _t = np.stack(_target, axis=0)

        # source and target should have the same shape
        if 'C' in _axes:
            # y doesn't have C channel
            if remove_C_dim(_s.shape, _axes) != _t.shape:
                raise ValueError('Shape of source and target images are incompatible.')
        else:
            # exact same shape
            if _s.shape != _t.shape:
                raise ValueError('Shape of source and target images are incompatible.')

        return _s, _t, _axes

    return _source, _target, axes


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
        for _, t in pairs:
            if not t.exists():
                raise FileNotFoundError(f'Could not find {t.absolute()}')
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


def load_from_disk(path, axes: str):
    """
    Load images from disk. If the dimensions don't agree, the method returns a tuple of list ([images], [files]). If
    the dimensions agree, the images are stacked along the `S` dimension of `axes` or along a new dimension if `S` is
    not in `axes`.

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

    if len(images) > 0 and dims_agree:
        if 'S' in axes:
            ind_S = axes.find('S')
            new_axes = axes
            final_images = np.concatenate(images, axis=ind_S)
        else:
            new_axes = 'S' + axes
            final_images = np.stack(images, axis=0)
        return final_images, new_axes

    return (images, image_files), axes


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
