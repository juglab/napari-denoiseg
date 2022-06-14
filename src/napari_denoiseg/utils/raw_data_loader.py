from pathlib import Path

import numpy as np
from csbdeep.data import RawData
from tifffile import imread
from csbdeep.utils import consume, axes_check_and_normalize


# TODO: should we keep basepath?
# Adapted from:
# https://csbdeep.bioimagecomputing.com/doc/_modules/csbdeep/data/rawdata.html#RawData.from_folder
def from_folder(basepath, source_dir, target_dir, axes='CZYX', check_exists=True):
    """
    Builds a generator for pairs of source and target images with same names. `check_exists` = `False` allows inserting
    empty images when the corresponding target is not found.

    Adapted from RawData.from_folder in CSBDeep.

    :param basepath: Parent directory for `source_dir` and `target_dir`
    :param source_dir: Relative folder containing source images
    :param target_dir: Relative folder containing target images, with same names than in `source_folder`
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
        Substitute the first element in tuple `ind` with `None`
        :param tuple_list: List of tuples
        :param ind: Index of the tuple in which to substitute the first element with `None`
        :return:
        """
        tuple_list[ind] = (tuple_list[ind][0], None)

    def _raise(e):
        raise e

    # pattern of images to select
    pattern = '*.tif*'

    # list of possible pairs based on the file found in the source folder
    p = Path(basepath)
    pairs = [(f, p / target_dir / f.name) for f in (p / source_dir).glob(pattern)]
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
    description = "{p}: target='{o}', sources={s}, axes='{a}', pattern='{pt}'".format(p=basepath,
                                                                                      s=source_dir,
                                                                                      o=target_dir, a=axes,
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
