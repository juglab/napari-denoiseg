import numpy as np
from napari.qt.threading import thread_worker
from napari_denoiseg.utils import UpdateType, State, generate_config, reshape_data_single


@thread_worker(start_thread=False)
def prediction_worker(widget):
    from denoiseg.models import DenoiSeg
    from napari_denoiseg.utils import load_from_disk, load_weights

    # from disk, lazy loading and threshold
    is_from_disk = widget.load_from_disk
    is_threshold = widget.threshold_cbox.isChecked()
    is_lazy_loading = widget.lazy_loading.isChecked()

    # get axes
    axes = widget.axes_widget.get_axes()

    # grab images
    if is_from_disk:
        images = load_from_disk(widget.images_folder.get_folder())
    else:
        images = widget.images.value.data
    assert len(images.shape) > 0

    # reshape data
    x, new_axes = reshape_data_single(images, axes)

    # yield total number of images
    n_img = images.shape[0]
    yield {UpdateType.N_IMAGES: n_img}

    # instantiate model with dummy values
    if 'Z' in new_axes:
        patch = (16, 16, 16)
    else:
        patch = (16, 16)
    config = generate_config(images, patch, 1, 1, 1)
    model = DenoiSeg(config, 'DenoiSeg', 'models')

    # this is to prevent the memory from saturating on the gpu on my machine
    # if tf.config.list_physical_devices('GPU'):
    #    tf.config.experimental.set_memory_growth(tf.config.list_physical_devices('GPU')[0], True)

    # set the weights of the model
    weight_name = widget.load_button.Model.value
    assert len(weight_name.name) > 0, 'Model path cannot be empty.'
    load_weights(model, weight_name)

    # loop over slices
    for i in range(n_img):
        # yield image number + 1
        yield {UpdateType.IMAGE: i + 1}

        # predict
        # TODO: axes make sure it is compatible with time, channel, z
        pred = model.predict(images[np.newaxis, i, :, :, np.newaxis], axes='SYXC')

        # threshold the foreground probability map
        pred_seg = pred[0, :, :, 2] >= widget.threshold_spin.Threshold.value

        # add prediction to layers
        widget.seg_prediction[i, :, :] = pred_seg
        widget.denoi_prediction[i, :, :] = pred[0, :, :, 0]

        # TODO: show also the border class

        # check if stop requested
        if widget.state != State.RUNNING:
            break

    # update done
    yield {UpdateType.DONE}
