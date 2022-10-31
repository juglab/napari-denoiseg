import numpy as np
from napari.qt.threading import thread_worker
from napari.utils import notifications as ntf
from denoiseg.utils.seg_utils import convert_to_oneHot
from napari_denoiseg.utils import (
    load_pairs_from_disk,
    reshape_data,
    load_model,
    State
)

from denoiseg.utils.compute_precision_threshold import measure_precision, compute_labels


@thread_worker(start_thread=False)
def optimizer_worker(widget):
    # loading images from disk?
    is_from_disk = widget.load_from_disk

    # get axes
    axes = widget.axes_widget.get_axes()

    # grab images
    # TODO here can be list of files
    if is_from_disk:
        try:
            images, labels, new_axes = load_pairs_from_disk(widget.images_folder.get_folder(),
                                                            widget.labels_folder.get_folder(),
                                                            axes)
        except FileNotFoundError as e:
            # ntf.show_error('Error loading images. Make sure they have the same name in each folder.')
            ntf.show_info('Error loading images. Make sure they have the same name in each folder.')
            print(e)
            return
    else:
        new_axes = axes
        images, labels = widget.images.value.data, widget.labels.value.data

    # reshape data
    _x, _y, new_axes = reshape_data(images, labels, new_axes)
    assert _x.shape[0] > 0
    assert _x.shape[:-1] == _y.shape, 'Image and labels should have the same shape (except C dimension)'

    # load model
    weight_path = widget.get_model_path()

    try:
        model = load_model(weight_path)
    except ValueError as e:
        # TODO: napari 0.4.16 has ntf.show_error, but napari workflows requires 0.4.15 that doesn't
        # ntf.show_error('Error loading model weights.')
        ntf.show_info('Error loading model weights.')
        print(e)
        return

    # threshold data to estimate the best threshold
    yield from optimize_threshold(model, _x, _y, new_axes, widget=widget)


def optimize_threshold(model, image_data, label_data, axes, widget=None):
    """

    :return:
    """
    # make sure that labels are int64
    lab_gt = label_data.astype(np.int64)
    image_32 = image_data.astype(np.float32)

    for i_t, ts in enumerate(np.linspace(0.1, 1, 19)):

        shape = (*image_data.shape[:-1],)
        scores = []
        for i_s in range(shape[0]):
            # predict and select only the segmentation predictions
            prediction = model.predict(image_32[i_s, ...], axes=axes[1:])[..., -3:]

            # compute labels
            lab_pred, _ = compute_labels(prediction, ts)

            score = measure_precision()(lab_gt[i_s], lab_pred)
            scores.append(score)

        if widget is not None and widget.state == State.IDLE:
            break

        yield i_t, ts, np.nanmean(scores)
