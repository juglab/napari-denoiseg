import os
from pathlib import Path

import numpy as np

from denoiseg.models import DenoiSeg


def generate_config(X,
                    patch_shape,
                    n_epochs=20,
                    n_steps=400,
                    batch_size=16,
                    **kwargs):
    from denoiseg.models import DenoiSegConfig

    # assert len(X.shape)-2 == len(patch_shape)
    # TODO: what if generator or list
    conf = DenoiSegConfig(X,
                          n_channel_out=X.shape[-1]+3,
                          n_channel_in=X.shape[-1],
                          train_steps_per_epoch=n_steps,
                          train_epochs=n_epochs,
                          batch_norm=True,
                          train_batch_size=batch_size,
                          n2v_patch_shape=patch_shape,
                          train_tensorboard=True,
                          **kwargs)

    return conf


def load_configuration(path):
    from csbdeep.utils import load_json
    from denoiseg.models import DenoiSegConfig

    # load config
    json_config = load_json(path)

    # create DenoiSeg configuration
    axes_length = len(json_config['axes'])
    n_channels = json_config['n_channel_in']

    if axes_length == 3:
        X = np.zeros((1, 8, 8, n_channels))
    else:
        X = np.zeros((1, 8, 8, 8, n_channels))

    return DenoiSegConfig(X, **json_config)


def save_configuration(config, dir_path):
    from csbdeep.utils import save_json

    # sanity check
    assert Path(dir_path).is_dir()

    # save
    final_path = Path(dir_path) / 'config.json'
    save_json(vars(config), final_path)


def load_weights(model: DenoiSeg, weights_path):
    """

    :param model:
    :param weights_path:
    :return:
    """
    _filename, file_ext = os.path.splitext(weights_path)
    if file_ext == ".zip":
        import bioimageio.core
        # we assume we got a modelzoo file
        rdf = bioimageio.core.load_resource_description(weights_path)
        weights_name = rdf.weights['keras_hdf5'].source
    else:
        # we assure we have a path to a .h5
        weights_name = weights_path

    if not Path(weights_name).exists():
        raise FileNotFoundError('Invalid path to weights.')

    model.keras_model.load_weights(weights_name)


def load_model(weight_path):

    if not Path(weight_path).exists():
        raise ValueError('Invalid model path.')

    if not (Path(weight_path).parent / 'config.json').exists():
        raise ValueError('No config.json file found.')

    # load configuration
    config = load_configuration(Path(weight_path).parent / 'config.json')

    # instantiate model
    model = DenoiSeg(config, 'DenoiSeg', 'models')

    # load weights
    load_weights(model, weight_path)

    return model
