import os
import warnings
from pathlib import Path

import numpy as np
from napari_time_slicer import time_slicer
from napari_tools_menu import register_function
from tensorflow.python.framework.errors_impl import UnknownError
from tifffile import imwrite

from napari.qt.threading import thread_worker
from napari.utils import notifications as ntf
from napari_denoiseg.utils import (
    UpdateType,
    State,
    reshape_data_single,
    load_model,
    reshape_napari
)


# todo: check the current menu items
# TODO: setup.cfg does not define an entry point currently
@register_function(menu="Segmentation / noise removal > Apply DenoiSeg")
@time_slicer
def apply_denoiseg(image: "napari.types.ImageData",
                   model_filename: os.PathLike = "my_denoiseg_model",
                   number_of_tiles: int = 4,
                   ) -> ("napari.types.ImageData", "napari.types.ImageData"):
    """
    """
    model_path = Path(model_filename)
    if not model_path.exists():
        raise Exception('Model not found')

    # load the model
    model = load_model(model_path)

    # check image shape
    if len(image.shape) == 2:
        axes = "YXC"
        tiles = (number_of_tiles, number_of_tiles, 1)
    elif len(image.shape) == 3:
        axes = "ZYXC"
        tiles = (number_of_tiles, number_of_tiles, number_of_tiles, 1)
    else:
        raise ValueError("Only 2D and 3D data supported.")

    # run prediction
    predicted_image = model.predict(image[..., np.newaxis], axes=axes, n_tiles=tiles)

    return predicted_image[..., :-3], predicted_image[..., -3:]


@thread_worker(start_thread=False)
def prediction_worker(widget):
    from napari_denoiseg.utils import load_from_disk, lazy_load_generator

    # from disk, lazy loading and threshold
    is_from_disk = widget.load_from_disk
    is_lazy_loading = widget.lazy_loading.isChecked()

    is_tiled = widget.tiling_cbox.isChecked()
    n_tiles = widget.tiling_spin.value()

    is_threshold = widget.threshold_cbox.isChecked()
    threshold = widget.threshold_spin.Threshold.value

    # get axes
    axes = widget.axes_widget.get_axes()

    # load model
    weight_path = widget.get_model_path()
    try:
        model = load_model(weight_path)
    except ValueError as e:
        # ntf.show_error('Error loading model weights.')
        ntf.show_info('Error loading model weights.')
        print(e)

        yield {UpdateType.DONE}
        return

    # grab images
    if is_from_disk:
        if is_lazy_loading:
            images, n_img = lazy_load_generator(widget.images_folder.get_folder())

            if n_img == 0:
                # ntf.show_error('No image found.')
                ntf.show_info('No image found.')
                yield {UpdateType.DONE}
                return

            new_axes = axes
        else:
            images, new_axes = load_from_disk(widget.images_folder.get_folder(), axes)

            if type(images) == tuple and len(images[0]) == 0:
                # ntf.show_error('No image found.')
                ntf.show_info('No image found.')
                yield {UpdateType.DONE}
                return
    else:
        images = widget.images.value.data
        new_axes = axes
        assert len(images.shape) > 0

    # common parameters list
    parameters = {'widget': widget,
                  'model': model,
                  'axes': new_axes,
                  'images': images,
                  'is_threshold': is_threshold,
                  'threshold': threshold,
                  'is_tiled': is_tiled,
                  'n_tiles': n_tiles}

    if is_from_disk and is_lazy_loading:
        # yield generator size
        yield {UpdateType.N_IMAGES: n_img}
        yield from _run_lazy_prediction(**parameters)
    elif is_from_disk and type(images) == tuple:  # load images from disk with different sizes
        yield from _run_prediction_to_disk(**parameters)
    else:
        yield from _run_prediction(**parameters)


def _run_prediction(widget,
                    model,
                    axes,
                    images,
                    is_threshold=False,
                    threshold=0.8,
                    is_tiled=False,
                    n_tiles=4):
    """
    Run prediction and show the result in napari.

    :param widget:
    :param model:
    :param axes:
    :param images: np.array
    :param is_threshold:
    :param threshold:
    :return:
    """

    # reshape data
    try:
        _data, new_axes = reshape_data_single(images, axes)
    except ValueError as e:
        msg = 'Data and axes are incompatible.'
        ntf.show_info(msg)
        warnings.warn(msg)
        print(e)
        return

    yield {UpdateType.N_IMAGES: _data.shape[0]}

    # this is to prevent the memory from saturating on the gpu on my machine
    # if tf.config.list_physical_devices('GPU'):
    #    tf.config.experimental.set_memory_growth(tf.config.list_physical_devices('GPU')[0], True)

    # TODO by replacing here predict_all by the widget.denoi and seg, we could show progress to users
    shape_out = (*_data.shape[:-1], _data.shape[-1] + 3)
    predict_all = np.zeros(shape_out, dtype=np.float32)

    # start predicting
    for i_slice in range(_data.shape[0]):
        # check if stop requested
        if widget.state != State.RUNNING:
            break

        # otherwise proceed
        _x = _data[np.newaxis, i_slice, ...]  # replace S dimension with singleton

        # yield image number + 1
        yield {UpdateType.IMAGE: i_slice + 1}

        # predict
        try:
            if is_tiled:
                predict_all[i_slice, ...] = model.predict(_x, axes=new_axes, n_tiles=n_tiles)
            else:
                predict_all[i_slice, ...] = model.predict(_x, axes=new_axes)
        except UnknownError as e:
            msg = 'UnknownError can be a failure to load cudnn, try restarting the computer.'
            # TODO: napari 0.4.16 has ntf.show_error, but napari workflows requires 0.4.15 that doesn't
            # ntf.show_error(msg)
            ntf.show_info(msg)
            warnings.warn(msg)
            print(e.message)
            break

    # split predictions
    final_image_d = predict_all[..., 0:-3].squeeze()
    final_image_s = predict_all[..., -3:].squeeze()

    if is_threshold:
        # if only one sample, then update new axes
        if predict_all.shape[0] == 1:
            new_axes = new_axes[1:]

        final_image_s = final_image_s >= threshold

        # Important: viewer.add_image seems to convert images to YX dims at the end, but not viewer.add_labels
        final_image_s, _ = reshape_napari(final_image_s, new_axes)

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
    Load all data in memory and run prediction, saving the result to the disk.

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
            try:
                _data, _axes = reshape_data_single(im, axes_order)

                counter = counter + 1
                yield _data, f, _axes, counter

            except ValueError:
                # ntf.show_error(f'Wrong image shapes {f.stem} {im.shape}')
                ntf.show_info(f'Wrong image shapes {f.stem} {im.shape}')

    gen = generator(images, axes)

    # TODO this is a weird way to use the generator to pass its total length
    yield {UpdateType.N_IMAGES: next(gen)}

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

            # shape prediction
            shape_out = (*_x.shape[:-1], _x.shape[-1] + 3)
            prediction = np.zeros(shape_out, dtype=np.float32)

            for i_s in range(_x.shape[0]):
                try:
                    if is_tiled:
                        prediction[i_s, ...] = model.predict(_x[i_s, ...], axes=new_axes[1:], n_tiles=n_tiles)
                    else:
                        prediction[i_s, ...] = model.predict(_x[i_s, ...], axes=new_axes[1:])
                except UnknownError as e:
                    msg = 'UnknownError can be a failure to load cudnn, try restarting the computer.'
                    # TODO: napari 0.4.16 has ntf.show_error, but napari workflows requires 0.4.15 that doesn't
                    # ntf.show_error(msg)
                    ntf.show_info(msg)
                    warnings.warn(msg)
                    print(e.message)
                    break

            # if only one sample, then update new axes
            if prediction.shape[0] == 1:
                new_axes = new_axes[1:]

            # split predictions
            final_image_d = prediction[..., 0:-3].squeeze()
            final_image_s = prediction[..., -3:].squeeze()

            if is_threshold:
                final_image_s = final_image_s >= threshold

            # Save napari with axes order (XY at the end) in case we want to reopen it
            final_image_s, _ = reshape_napari(final_image_s, new_axes)

            # save predictions
            parent = Path(file.parent, 'results')
            if not parent.exists():
                os.mkdir(parent)

            new_file_path_denoi = Path(parent, file.stem + '_denoised' + file.suffix)
            new_file_path_seg = Path(parent, file.stem + '_segmented' + file.suffix)
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
    while True:
        next_tuple = next(images, None)

        if next_tuple is not None:
            image, file, i_im = next_tuple

            yield {UpdateType.IMAGE: i_im}

            # reshape data
            try:
                _x, new_axes = reshape_data_single(image, axes)

                # run prediction
                shape_out = (*_x.shape[:-1], _x.shape[-1] + 3)
                prediction = np.zeros(shape_out, dtype=np.float32)

                for i_s in range(_x.shape[0]):
                    try:
                        if is_tiled:
                            prediction[i_s, ...] = model.predict(_x[i_s, ...], axes=new_axes[1:], n_tiles=n_tiles)
                        else:
                            prediction[i_s, ...] = model.predict(_x[i_s, ...], axes=new_axes[1:])
                    except UnknownError as e:
                        msg = 'UnknownError can be a failure to load cudnn, try restarting the computer.'
                        # TODO: napari 0.4.16 has ntf.show_error, but napari workflows requires 0.4.15 that doesn't
                        # ntf.show_error(msg)
                        ntf.show_info(msg)
                        warnings.warn(msg)
                        print(e.message)
                        break

                # if only one sample, then update new axes
                if prediction.shape[0] == 1:
                    new_axes = new_axes[1:]

                # split predictions
                final_image_d = prediction[..., 0:-3].squeeze()
                final_image_s = prediction[..., -3:].squeeze()

                if is_threshold:
                    final_image_s = final_image_s >= threshold

                # Save napari with axes order (XY at the end) in case we want to reopen it
                final_image_s, _ = reshape_napari(final_image_s, new_axes)

                # save predictions
                parent = Path(file.parent, 'results')
                if not parent.exists():
                    os.mkdir(parent)

                new_file_path_denoi = Path(parent, file.stem + '_denoised' + file.suffix)
                new_file_path_seg = Path(parent, file.stem + '_segmented' + file.suffix)
                imwrite(new_file_path_denoi, final_image_d)
                imwrite(new_file_path_seg, final_image_s)

                # check if stop requested
                if widget.state != State.RUNNING:
                    break

            except ValueError:
                ntf.show_info(f'Wrong image shapes  {file.stem} {image.shape}')
                # ntf.show_error(f'Wrong image shapes  {file.stem} {image.shape}')
        else:
            break

    # update done
    yield {UpdateType.DONE}
