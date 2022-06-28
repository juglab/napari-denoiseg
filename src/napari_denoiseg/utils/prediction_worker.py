import numpy as np
from napari.qt.threading import thread_worker
from napari_denoiseg.utils import generate_config, UpdateType, State


@thread_worker(start_thread=False)
def prediction_worker(widget):
    from denoiseg.models import DenoiSeg
    from napari_denoiseg.utils import load_from_disk, load_weights

    # grab images
    if widget.load_from_disk:
        images = load_from_disk(widget.images_folder.get_folder())
    else:
        images = widget.images.value.data
    assert len(images.shape) > 1

    # yield total number of images
    n_img = images.shape[0]
    yield {UpdateType.N_IMAGES: n_img}

    # set extra dimensions
    images = images[np.newaxis,... , np.newaxis]

    images = np.array(images)
    # instantiate model with dummy values
    config = generate_config(images, tuple([1 for x in range(len(images.shape)-2)]), 1, 1, 1)
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

        # check if stop requested
        if widget.state != State.RUNNING:
            break

    # update done
    yield {UpdateType.DONE}
