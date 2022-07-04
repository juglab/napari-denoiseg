from pathlib import Path

import numpy as np
from tifffile import imwrite

from napari.qt.threading import thread_worker
from napari_denoiseg.utils import UpdateType, State, generate_config, reshape_data_single


@thread_worker(start_thread=False)
def prediction_worker(widget):
    from denoiseg.models import DenoiSeg
    from napari_denoiseg.utils import load_from_disk, lazy_load_generator, load_weights

    # from disk, lazy loading and threshold
    is_from_disk = widget.load_from_disk
    is_lazy_loading = widget.lazy_loading.isChecked()
    is_threshold = widget.threshold_cbox.isChecked()
    threshold = widget.threshold_spin.Threshold.value

    # get axes
    axes = widget.axes_widget.get_axes()

    # grab images
    if is_from_disk:
        if is_lazy_loading:
            images, n_img = lazy_load_generator(widget.images_folder.get_folder())
            assert n_img > 0
        else:
            images = load_from_disk(widget.images_folder.get_folder(), axes)
            assert len(images.shape) > 0
    else:
        images = widget.images.value.data
        assert len(images.shape) > 0

    # instantiate model with dummy values
    if 'Z' in axes:
        patch = (16, 16, 16)
    else:
        patch = (16, 16)
    config = generate_config(images, patch, 1, 1, 1)  # TODO: images in lazy load will not work here
    model = DenoiSeg(config, 'DenoiSeg', 'models')

    # this is to prevent the memory from saturating on the gpu on my machine
    # if tf.config.list_physical_devices('GPU'):
    #    tf.config.experimental.set_memory_growth(tf.config.list_physical_devices('GPU')[0], True)

    # load model weights
    weight_name = widget.load_button.Model.value
    assert len(weight_name.name) > 0, 'Model path cannot be empty.'
    load_weights(model, weight_name)

    if is_from_disk and is_lazy_loading:
        # yield generator size
        yield {UpdateType.N_IMAGES: n_img}
        yield from _run_lazy_prediction(widget, model, axes, images, is_threshold, threshold)
    else:
        yield from _run_prediction(widget, model, axes, images, is_threshold, threshold)


def _run_prediction(widget, model, axes, images, is_threshold=False, threshold=0.8):
    """

    :param widget:
    :param model:
    :param axes:
    :param images:
    :param is_threshold:
    :param threshold:
    :return:
    """
    def generator(data, axes_order):
        """

        :param data:
        :param axes_order:
        :return:
        """
        if type(data) == list:
            yield len(data)
            for j, d in enumerate(data):
                _data, _axes = reshape_data_single(d, axes_order)
                yield _data, _axes, j
        else:
            _data, _axes = reshape_data_single(data, axes_order)
            yield _data.shape[0]

            for k in range(_data.shape[0]):
                yield _data[np.newaxis, k, ...], _axes, k

    gen = generator(images, axes)
    n_img = next(gen)
    yield {UpdateType.N_IMAGES: n_img}

    while True:
        t = next(gen)

        if t is not None:
            _x, new_axes, i = t

            # yield image number + 1
            yield {UpdateType.IMAGE: i + 1}

            # predict
            prediction = model.predict(_x, axes=new_axes)

            # split predictions and threshold if requested
            # TODO does this work with napari layers? YX dims at the end
            if is_threshold:
                denoised = prediction[0, ..., 0:-3]  # denoised channels
                segmented = prediction[0, ..., -3:] >= threshold
            else:
                denoised = prediction[0, ..., 0:-3]
                segmented = prediction[0, ..., -3:]

            # update the layers in napari
            widget.seg_prediction[i, ...] = segmented
            widget.denoi_prediction[i, ...] = denoised

            # check if stop requested
            if widget.state != State.RUNNING:
                break

    # update done
    yield {UpdateType.DONE}


def _run_lazy_prediction(widget, model, axes, generator, is_threshold=False, threshold=0.8):
    while True:
        next_tuple = next(generator, None)

        if next_tuple is not None:
            image, file, i = next_tuple

            yield {UpdateType.IMAGE: i}

            # reshape data
            x, new_axes = reshape_data_single(image, axes)

            # run prediction
            prediction = model.predict(x, axes=new_axes)

            # split predictions and threshold if requested
            # TODO does this work with napari layers? YX dims at the end
            if is_threshold:
                denoised = prediction[0, ..., 0:-3]  # denoised channels
                segmented = prediction[0, ..., -3:] >= threshold
            else:
                denoised = prediction[0, ..., 0:-3]
                segmented = prediction[0, ..., -3:]

            # save predictions
            new_file_path_denoi = Path(file.parent, file.stem + '_denoised' + file.suffix)
            new_file_path_seg = Path(file.parent, file.stem + '_segmented' + file.suffix)
            imwrite(new_file_path_denoi, denoised)
            imwrite(new_file_path_seg, segmented)

            # check if stop requested
            if widget.state != State.RUNNING:
                break
        else:
            break

    # update done
    yield {UpdateType.DONE}
