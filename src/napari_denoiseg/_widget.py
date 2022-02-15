"""
This module is an example of a barebones QWidget plugin for napari

It implements the Widget specification.
see: https://napari.org/plugins/stable/guides.html#widgets

Replace code below according to your needs.
"""
from napari.qt.threading import thread_worker
from magicgui import magic_factory, widgets
import time


# "step 50/200 25%"

@magic_factory(perc_train_labels={"widget_type": "FloatSlider", "min": 0.1, "max": 1., "step": 0.05, 'value': 0.6},
               n_epochs={"widget_type": "SpinBox", "step": 1, 'value': 10},
               n_steps={"widget_type": "SpinBox", "step": 1, 'value': 200},
               patch_shape={"widget_type": "Slider", "min": 16, "max": 512, "step": 16, 'value': 64},
               batch_size={"widget_type": "Slider", "min": 16, "max": 512, "step": 16, 'value': 64},
               neighborhood_radius={"widget_type": "Slider", "min": 1, "max": 16, 'value': 16},
               epoch_prog={'visible': True, 'min': 0, 'max': 100, 'step': 1, 'value': 0, 'label': 'epochs'},
               step_prog={'visible': True, 'min': 0, 'max': 100, 'step': 1, 'value': 0, 'label': 'steps'})
def denoiseg_widget(data: 'napari.layers.Image',
                    ground_truth: 'napari.layers.Labels',
                    perc_train_labels: float,
                    n_epochs: int,
                    n_steps: int,
                    patch_shape: int,
                    batch_size: int,
                    neighborhood_radius: int,
                    epoch_prog: widgets.ProgressBar,
                    step_prog: widgets.ProgressBar):
    def update_progress(update):
        epoch_prog.native.setValue(update[0])
        step_prog.native.setValue(update[1])

    # ProgressBar.native.setValue avoids bug in magicgui ProgressBar.increment(val)
    # @thread_worker(connect={'yielded': epoch_prog.native.setValue})
    @thread_worker(connect={'yielded': update_progress})
    def process():
        n_pt = 5
        n_sp = 10
        t_s = 0.1
        for i in range(n_pt):
            p_epoch = int(i * 100 / (n_pt - 1) + 0.5)

            for j in range(n_sp):
                p_step = int(j * 100 / (n_sp - 1) + 0.5)

                # update progress bar
                yield p_epoch, p_step

                # sleep
                time.sleep(t_s)

    print(f'Data shape {data.data.shape}')
    print(f'Total number of labels {ground_truth.data.shape[0]}')

    n_labels = int(0.5 + perc_train_labels * ground_truth.data.shape[0])
    print(f'Number of labels used for training {n_labels}')

    # split train and val
    X, Y, X_val, Y_val = __prepare_data(data.data, ground_truth.data, perc_train_labels)

    config = {'n_epochs': n_epochs,
              'n_steps': n_steps,
              'patch_shape': patch_shape,
              'batch_size': batch_size,
              'neighborhood_radius': neighborhood_radius}
    print(config)

    process()


def __prepare_data(data, gt, perc_labels):
    import numpy as np
    from denoiseg.utils.misc_utils import augment_data
    from denoiseg.utils.seg_utils import convert_to_oneHot

    def zero_sum(im):
        im_reshaped = im.reshape(im.shape[0], -1)

        return np.where(np.sum(im_reshaped != 0, axis=1) != 0)[0]

    def list_diff(l1, l2):
        return list(set(l1) - set(l2))

    def create_train_set(x, y, ind_exclude):
        masks = np.zeros(x.shape)
        masks[0:y.shape[0], 0:y.shape[1], 0:y.shape[2]] = y  # there's probably a more elegant way

        return augment_data(np.delete(x, ind_exclude, axis=0), np.delete(masks, ind_exclude, axis=0))

    def create_val_set(x, y, ind_include):
        return np.take(x, ind_include, axis=0), np.take(y, ind_include, axis=0)

    # currently: no shuffling. we detect the non-empty labeled frames and split them. The validation set is
    # entirely constituted of the frames corresponding to the split labeled frames, the training set is all the
    # remaining images. The training gt is then the remaining labeled frames and empty frames.
    
    # get indices of labeled frames
    n_labels = int(0.5 + perc_labels * gt.data.shape[0])
    ind = zero_sum(gt)
    assert n_labels < len(ind)

    # split labeled frames between train and val sets
    ind_train = np.random.choice(ind, size=n_labels, replace=False).tolist()
    ind_val = list_diff(ind, ind_train)
    assert len(ind_train) + len(ind_val) == len(ind)

    # create train and val sets
    x_train, y_train = create_train_set(data, gt, ind_val)
    x_val, y_val = create_val_set(data, gt, ind_val)

    # add channel dim and one-hot encoding
    X = x_train[..., np.newaxis]
    Y = convert_to_oneHot(y_train)
    X_val = x_val[..., np.newaxis]
    Y_val = convert_to_oneHot(y_val)

    return X, Y, X_val, Y_val


def __train():
    pass
