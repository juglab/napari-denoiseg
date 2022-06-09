from pathlib import Path
from itertools import chain

import numpy as np
from csbdeep.data import RawData
from tifffile import imread
from csbdeep.utils import _raise, consume, axes_check_and_normalize


# https://csbdeep.bioimagecomputing.com/doc/_modules/csbdeep/data/rawdata.html#RawData.from_folder
def from_folder(basepath, source_dir, target_dir, axes='CZYX', pattern='*.tif*', check_exists=True):
    def substitute_by_none(tuple_list, ind):
        tuple_list[ind] = (tuple_list[ind][0], None)

    p = Path(basepath)
    pairs = [(f, p / target_dir / f.name) for f in (p / source_dir).glob(pattern)]
    len(pairs) > 0 or _raise(FileNotFoundError("Didn't find any images."))

    if check_exists:
        consume(t.exists() or _raise(FileNotFoundError(t)) for s, t in pairs)
    else:
        consume(p[1].exists() or substitute_by_none(pairs, i) for i, p in enumerate(pairs))

    axes = axes_check_and_normalize(axes)
    n_images = len(pairs)
    description = "{p}: target='{o}', sources={s}, axes='{a}', pattern='{pt}'".format(p=basepath,
                                                                                      s=source_dir,
                                                                                      o=target_dir, a=axes,
                                                                                      pt=pattern)

    def _gen():
        for fx, fy in pairs:
            if fy:
                x, y = imread(str(fx)), imread(str(fy))
            else:
                x = imread(str(fx))
                y = np.zeros(x.shape)

            len(axes) >= x.ndim or _raise(ValueError())
            yield x, y, axes[-x.ndim:], None

    return RawData(_gen, n_images, description)
