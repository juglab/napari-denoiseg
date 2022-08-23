from pathlib import Path

import numpy as np
from tifffile import imwrite

from napari.qt.threading import thread_worker
from napari_denoiseg.utils import (
    UpdateType,
    State,
    reshape_data_single,
    load_model,
    reshape_napari,
    get_napari_shapes
)


@thread_worker(start_thread=False)
def prediction_worker(widget):
    from napari_denoiseg.utils import load_from_disk, lazy_load_generator

    # from disk, lazy loading and threshold
    is_from_disk = widget.load_from_disk
    is_lazy_loading = widget.lazy_loading.isChecked()

    is_tiled = widget.tiling_cbox.isChecked()
    n_tiles = widget.tiling_spin.value

    is_threshold = widget.threshold_cbox.isChecked()
    threshold = widget.threshold_spin.Threshold.value

    # get axes
    axes = widget.axes_widget.get_axes()

    # load model
    weight_path = widget.get_model_path()
    model = load_model(weight_path)

    # grab images
    if is_from_disk:
        if is_lazy_loading:
            images, n_img = lazy_load_generator(widget.images_folder.get_folder())
            assert n_img > 0, 'No image returned.'
        else:
            images = load_from_disk(widget.images_folder.get_folder(), axes)
            assert len(images.shape) > 0
    else:
        images = widget.images.value.data
        assert len(images.shape) > 0

    # common parameters list
    parameters = {'widget': widget,
                  'model': model,
                  'axes': axes,
                  'is_threshold': is_threshold,
                  'threshold': threshold,
                  'is_tiled': is_tiled,
                  'n_tiles': n_tiles}

    if is_from_disk and is_lazy_loading:
        # yield generator size
        yield {UpdateType.N_IMAGES: n_img}
        yield from _run_lazy_prediction(**parameters, generator=images)
    elif is_from_disk and type(images) == tuple:
        yield from _run_prediction_to_disk(**parameters, images=images)
    else:
        # TODO: check if is_from_disk=True and images not a tuple is possible, otherwise is_from_disk doesn't have to be
        #  passed as parameter. Then we can simply have the same method signature for all three
        yield from _run_prediction(**parameters, images=images, is_from_disk=is_from_disk)


def _run_prediction(widget,
                    model,
                    axes,
                    images,
                    is_from_disk,
                    is_threshold=False,
                    threshold=0.8,
                    is_tiled=False,
                    n_tiles=4):
    """

    :param widget:
    :param model:
    :param axes:
    :param images: np.array
    :param is_threshold:
    :param threshold:
    :return:
    """

    # reshape data
    _data, new_axes = reshape_data_single(images, axes)
    yield {UpdateType.N_IMAGES: _data.shape[0]}

    # if the images were loaded from disk, the layers in napari have the wrong shape
    if is_from_disk:
        shape_denoised, shape_segmented = get_napari_shapes(images.shape, axes)
        widget.denoi_prediction = np.zeros(shape_denoised, dtype=np.float32)
        widget.seg_prediction = np.zeros(shape_segmented, dtype=widget.seg_prediction.dtype)

    # this is to prevent the memory from saturating on the gpu on my machine
    # if tf.config.list_physical_devices('GPU'):
    #    tf.config.experimental.set_memory_growth(tf.config.list_physical_devices('GPU')[0], True)

    final_image_d = np.zeros(_data.shape, dtype=np.float32).squeeze()
    final_image_s = np.zeros((*_data.shape[:-1], 3), dtype=np.float32).squeeze()

    # start predicting
    for i_slice in range(_data.shape[0]):
        _x = _data[np.newaxis, i_slice, ...]  # replace S dimension with singleton

        # yield image number + 1
        yield {UpdateType.IMAGE: i_slice + 1}

        # TODO refactor the separation between denoised and segmented into a testable function
        # predict
        if is_tiled:
            prediction = model.predict(_x, axes=new_axes, n_tiles=n_tiles)
        else:
            prediction = model.predict(_x, axes=new_axes)

        # split predictions and update the layers in napari
        final_image_d[i_slice, ...] = prediction[0, ..., 0:-3].squeeze()
        final_image_s[i_slice, ...] = prediction[0, ..., -3:].squeeze()

        # check if stop requested
        if widget.state != State.RUNNING:
            break

    if is_threshold:
        final_image_s_t = final_image_s >= threshold

        # Important: viewer.add_image seems to convert images to YX dims at the end, but not viewer.add_labels
        final_image_s, _ = reshape_napari(final_image_s_t, new_axes)

    widget.seg_prediction = final_image_s
    widget.denoi_prediction = final_image_d

    # update done
    yield {UpdateType.DONE}


def _run_prediction_to_disk(widget,
                            model,
                            axes,
                            images,
                            is_threshold=False,
                            threshold=0.8,
                            is_tiled=False,
                            n_tiles=4):
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
        yield len(data[0])
        counter = 0
        for im, f in zip(*data):
            # reshape from napari to S(Z)YXC
            _data, _axes = reshape_data_single(im, axes_order)
            counter += counter + 1
            yield _data, f, _axes, counter

    gen = generator(images, axes)
    n_img = next(gen)
    yield {UpdateType.N_IMAGES: n_img}

    # this is to prevent the memory from saturating on the gpu on my machine
    # if tf.config.list_physical_devices('GPU'):
    #    tf.config.experimental.set_memory_growth(tf.config.list_physical_devices('GPU')[0], True)

    # start predicting
    while True:
        t = next(gen, None)

        if t is not None:
            _x, file, new_axes, i = t

            # yield image number
            yield {UpdateType.IMAGE: i}

            # TODO refactor the separation between denoised and segmented into a testable function
            # predict
            if is_tiled:
                prediction = model.predict(_x, axes=new_axes, n_tiles=n_tiles)
            else:
                prediction = model.predict(_x, axes=new_axes)

            # split predictions and threshold if requested
            final_image_d = prediction[0, ..., 0:-3].squeeze()
            final_image_s = prediction[0, ..., -3:].squeeze()

            if is_threshold:
                final_image_s_t = final_image_s >= threshold

            # Save napari axes order (XY at the end) in case we want to reopen it
            final_image_s, _ = reshape_napari(final_image_s_t, new_axes)

            # save predictions
            new_file_path_denoi = Path(file.parent, file.stem + '_denoised' + file.suffix)
            new_file_path_seg = Path(file.parent, file.stem + '_segmented' + file.suffix)
            imwrite(new_file_path_denoi, final_image_d)
            imwrite(new_file_path_seg, final_image_s)

            # check if stop requested
            if widget.state != State.RUNNING:
                break
        else:
            break

    # update done
    yield {UpdateType.DONE}


def _run_lazy_prediction(widget,
                         model,
                         axes,
                         generator,
                         is_threshold=False,
                         threshold=0.8,
                         is_tiled=False,
                         n_tiles=4):
    """

    :param widget:
    :param model:
    :param axes:
    :param generator:
    :param is_threshold:
    :param threshold:
    :return:
    """
    while True:
        next_tuple = next(generator, None)

        if next_tuple is not None:
            image, file, i_im = next_tuple

            yield {UpdateType.IMAGE: i_im}

            # reshape data
            _x, new_axes = reshape_data_single(image, axes)

            # run prediction
            shape_out = (*_x.shape[:-1], _x.shape[-1] + 3)
            prediction = np.zeros(shape_out, dtype=np.float32)

            # TODO: why can't we predict all S together? csbdeep throws error for axes and dims mismatch
            for i_s in range(_x.shape[0]):
                if is_tiled:
                    prediction[i_s, ...] = model.predict(_x[i_s, ...], axes=new_axes[1:], n_tiles=n_tiles)
                else:
                    prediction[i_s, ...] = model.predict(_x[i_s, ...], axes=new_axes[1:])

            # if only one sample, then update new axes
            if prediction.shape[0] == 1:
                new_axes = new_axes[1:]

            # split predictions
            final_image_d = prediction[..., 0:-3].squeeze()
            final_image_s = prediction[..., -3:].squeeze()

            if is_threshold:
                final_image_s_t = final_image_s >= threshold
            else:
                final_image_s_t = final_image_s

            # Save napari with axes order (XY at the end) in case we want to reopen it
            final_image_s, _ = reshape_napari(final_image_s_t, new_axes)

            # save predictions
            new_file_path_denoi = Path(file.parent, file.stem + '_denoised' + file.suffix)
            new_file_path_seg = Path(file.parent, file.stem + '_segmented' + file.suffix)
            imwrite(new_file_path_denoi, final_image_d)
            imwrite(new_file_path_seg, final_image_s)

            # check if stop requested
            if widget.state != State.RUNNING:
                break
        else:
            break

    # update done
    yield {UpdateType.DONE}
