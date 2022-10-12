from napari.qt.threading import thread_worker
from denoiseg.utils.seg_utils import convert_to_oneHot
from napari_denoiseg.utils import (
    load_pairs_from_disk,
    optimize_threshold,
    reshape_data,
    load_model
)


@thread_worker(start_thread=False)
def optimizer_worker(widget):
    # loading images from disk?
    is_from_disk = widget.load_from_disk

    # get axes
    axes = widget.axes_widget.get_axes()

    # grab images
    if is_from_disk:
        images, labels, _ = load_pairs_from_disk(widget.images_folder.get_folder(),
                                                 widget.labels_folder.get_folder(),
                                                 axes)
        # TODO should return with message to the user here if this method returns a list of images
    else:
        images, labels = widget.images.value.data, widget.labels.value.data

    # reshape data
    _x, _y, new_axes = reshape_data(images, labels, axes)
    assert _x.shape[0] > 0
    assert _x.shape[:-1] == _y.shape, 'Image and labels should have the same shape (except C dimension)'

    # convert to onehot
    _y_onehot = convert_to_oneHot(_y)

    # load model
    weight_path = widget.get_model_path()
    model = load_model(weight_path)

    # threshold data to estimate the best threshold
    yield from optimize_threshold(model, _x, _y_onehot, new_axes, widget=widget)
