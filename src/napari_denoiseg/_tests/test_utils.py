import os
from pathlib import Path
from tifffile import imwrite
import numpy as np
from denoiseg.models import DenoiSeg
from napari_denoiseg.utils import generate_config


###################################################################
# convenience functions: save images
def save_img(folder_path, n, shape, prefix=''):
    for i in range(n):
        im = np.random.randint(0, 65535, shape, dtype=np.uint16)
        imwrite(os.path.join(folder_path, prefix + str(i) + '.tif'), im)


def create_data(main_dir, folders, sizes, shape):
    for n, f in zip(sizes, folders):
        source = main_dir / f
        os.mkdir(source)
        save_img(source, n, shape)


def test_create_data(tmp_path):
    folders = ['train_X', 'train_Y', 'val_X', 'val_Y']
    sizes = [20, 8, 5, 5]

    create_data(tmp_path, folders, sizes, (1, 8, 16, 16))

    for n, f in zip(sizes, folders):
        source = tmp_path / f
        files = [f for f in Path(source).glob('*.tif*')]
        assert len(files) == n


# convenience functions: create and save models
def create_model(basedir, shape):
    # create model
    X = np.zeros(shape)
    name = 'myModel'
    config = generate_config(X)

    return DenoiSeg(config, name, basedir)


def save_weights_h5(model, basedir):
    name_weights = 'myModel.h5'
    path_to_weights = basedir / name_weights

    # save model
    model.keras_model.save_weights(path_to_weights)

    return path_to_weights


def test_saved_weights_h5(tmp_path):
    model = create_model(tmp_path, (1, 16, 16, 1))
    path_to_weights = save_weights_h5(model, tmp_path)

    assert path_to_weights.name.endswith('.h5')
    assert path_to_weights.exists()


# TODO: why is it saving in the current directory and not in folder?
def create_model_zoo_parameters(folder, shape):
    # create model and save it to disk
    model = create_model(folder, shape)
    path_to_h5 = str(save_weights_h5(model, folder).absolute())

    # path to modelzoo
    path_to_modelzoo = path_to_h5[:-len('.h5')] + '.bioimage.io.zip'

    # inputs/outputs
    path_to_input = path_to_h5[:-len('.h5')] + '-input.npy'
    np.save(path_to_input, np.zeros(shape))
    assert Path(path_to_input).exists()

    path_to_output = path_to_h5[:-len('.h5')] + '-output.npy'
    np.save(path_to_output, np.zeros(shape))
    assert Path(path_to_output).exists()

    # documentation
    path_to_doc = folder / 'doc.md'
    with open(path_to_doc, 'w') as f:
        pass
    assert path_to_doc.exists()

    # tf version
    tf_version = 42

    return path_to_modelzoo, path_to_h5, path_to_input, path_to_output, tf_version, path_to_doc
