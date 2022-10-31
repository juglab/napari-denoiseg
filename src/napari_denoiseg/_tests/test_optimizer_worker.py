import pytest

import numpy as np
from napari_denoiseg._tests.test_utils import (
    create_simple_model
)
from napari_denoiseg.utils import optimize_threshold


@pytest.mark.parametrize('shape, axes', [((1, 16, 16, 1), 'SYXC')])
def test_optimize_threshold(tmp_path, shape, axes):
    from denoiseg.utils.seg_utils import convert_to_oneHot

    # create model and data
    model = create_simple_model(tmp_path, shape)
    X_val = np.random.random(shape)

    if 'C' in axes:
        y_shape = shape[:-1]
    else:
        y_shape = shape

    Y_val = np.random.randint(0, 255, y_shape, dtype=np.int16)

    # instantiate generator
    gen = optimize_threshold(model, X_val, Y_val, axes)

    thresholds = []
    while True:
        t = next(gen, None)

        if t:
            _, temp_threshold, temp_score = t

            thresholds.append(temp_threshold)
        else:
            break

    assert len(thresholds) == 19
