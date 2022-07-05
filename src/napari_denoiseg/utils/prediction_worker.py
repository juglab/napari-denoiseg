from pathlib import Path

import numpy as np
from tifffile import imwrite
from denoiseg.models import DenoiSeg

from napari.qt.threading import thread_worker
from napari_denoiseg.utils import UpdateType, State, generate_config, reshape_data_single, load_weights


# TODO: Because of the loading yielding np.array, list or generator, the model is now instantiated in the prediction
#  loop. We might need to revisit this, as it is not very elegant.


@thread_worker(start_thread=False)
def prediction_worker(widget):
    from napari_denoiseg.utils import load_from_disk, lazy_load_generator

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

    if is_from_disk and is_lazy_loading:
        # yield generator size
        yield {UpdateType.N_IMAGES: n_img}
        yield from _run_lazy_prediction(widget, axes, images, is_threshold, threshold)
    else:
        yield from _run_prediction(widget, axes, images, is_threshold, threshold)


def _run_prediction(widget, axes, images, is_threshold=False, threshold=0.8):
    """

    :param widget:
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

    # instantiate model with dummy values
    if 'Z' in axes:
        patch = (16, 16, 16)
    else:
        patch = (16, 16)

    is_list = False
    if type(images) == list:
        is_list = True
        config = generate_config(images[0], patch, 1, 1, 1)
    else:
        config = generate_config(images, patch, 1, 1, 1)

    model = DenoiSeg(config, 'DenoiSeg', 'models')

    # this is to prevent the memory from saturating on the gpu on my machine
    # if tf.config.list_physical_devices('GPU'):
    #    tf.config.experimental.set_memory_growth(tf.config.list_physical_devices('GPU')[0], True)

    # load model weights
    weight_path = widget.get_model_path()
    if not Path(weight_path).exists():
        raise ValueError('Invalid model path.')

    load_weights(model, weight_path)

    # start predicting
    while True:
        t = next(gen)

        if t is not None:
            _x, new_axes, i = t

            # yield image number + 1
            yield {UpdateType.IMAGE: i + 1}

            # update std and mean
            if is_list:
                model.config = generate_config(_x, patch, 1, 1, 1)

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


def _run_lazy_prediction(widget, axes, generator, is_threshold=False, threshold=0.8):
    # instantiate model with dummy values
    if 'Z' in axes:
        patch = (16, 16, 16)
    else:
        patch = (16, 16)

    config, model = None, None
    while True:
        next_tuple = next(generator, None)

        if next_tuple is not None:
            image, file, i = next_tuple

            yield {UpdateType.IMAGE: i}

            # TODO: stupid to instantiate model there, update DenoiSeg to not need images to instantiate model?
            if i == 1:
                # instantiate model
                config = generate_config(image, patch, 1, 1, 1)
                model = DenoiSeg(config, 'DenoiSeg', 'models')

                # load model weights
                weight_path = widget.get_model_path()
                if not Path(weight_path).exists():
                    raise ValueError('Invalid model path.')

                load_weights(model, weight_path)
            else:
                model.config = generate_config(image, patch, 1, 1, 1)

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
