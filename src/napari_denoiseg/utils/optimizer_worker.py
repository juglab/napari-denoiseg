from napari.qt.threading import thread_worker
from napari_denoiseg.utils import generate_config, load_pairs_from_disk, load_weights, State
from utils.denoiseg_utils import optimize_threshold, load_pairs_generator, reshape_data


@thread_worker(start_thread=False)
def optimizer_worker(widget):
    from denoiseg.models import DenoiSeg

    # loading images from disk?
    is_from_disk = widget.load_from_disk

    # get axes
    axes = widget.axes_widget.get_axes()

    # grab images
    if is_from_disk:
        images, labels, _ = load_pairs_from_disk(widget.images_folder.get_folder(),
                                                 widget.labels_folder.get_folder(),
                                                 axes)
    else:
        images, labels = widget.images.value.data, widget.labels.value.data

    # reshape data
    _x, _y, new_axes = reshape_data(images, labels, axes)
    assert len(_x.shape[0]) > 0
    assert _x.shape[:-1] == _y.shape[:-1]  # exclude channels

    # instantiate model with dummy values
    if 'Z' in new_axes:
        patch = (16, 16, 16)
    else:
        patch = (16, 16)
    config = generate_config(_x, patch, 1, 1, 1)
    model = DenoiSeg(config, 'DenoiSeg', 'models')

    # load model weights
    weight_name = widget.load_button.Model.value
    assert len(weight_name.name) > 0, 'Model path cannot be empty.'
    load_weights(model, weight_name)

    # threshold data to estimate the best threshold
    yield from optimize_threshold(model, _x, _y, widget)
