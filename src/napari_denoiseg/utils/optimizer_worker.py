from pathlib import Path

import numpy as np
from napari.qt.threading import thread_worker
from napari_denoiseg.utils import generate_config, load_pairs_from_disk, load_weights, State


@thread_worker(start_thread=False)
def optimizer_worker(widget):
    from denoiseg.models import DenoiSeg
    from denoiseg.utils.compute_precision_threshold import measure_precision

    # get images
    if widget.load_from_disk:
        images = widget.images_folder.get_folder()
        assert Path(images.exists()), 'Images path does not exists.'

        labels = widget.labels_folder.get_folder()
        assert Path(labels.exists()), 'Labels path does not exists.'

        image_data, label_data = load_pairs_from_disk(images, labels)
    else:
        image_data = widget.images.value.data
        label_data = widget.labels.value.data
    assert image_data.shape == label_data.shape, 'Images and labels must have same shape.'

    # instantiate model
    config = generate_config(image_data[np.newaxis, 0, ..., np.newaxis], 1, 1, 1)  # TODO what if model won't fit?
    basedir = 'models'

    weights_name = widget.load_button.Model.value
    assert len(weights_name.name) > 0
    name = weights_name.stem

    # create model
    model = DenoiSeg(config, name, basedir)
    load_weights(model, weights_name)

    # threshold data to estimate the best threshold
    for i, ts in enumerate(np.linspace(0.1, 1, 19)):
        _, score = model.predict_label_masks(image_data, label_data, ts, measure_precision())

        # check if stop requested
        if widget.state != State.RUNNING:
            break

        yield i, ts, score

