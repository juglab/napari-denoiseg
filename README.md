# napari-denoiseg

[![License](https://img.shields.io/pypi/l/napari-denoiseg.svg?color=green)](https://github.com/juglab/napari_denoiseg/raw/main/LICENSE)
[![PyPI](https://img.shields.io/pypi/v/napari-denoiseg.svg?color=green)](https://pypi.org/project/napari-denoiseg)
[![Python Version](https://img.shields.io/pypi/pyversions/napari-denoiseg.svg?color=green)](https://python.org)
[![tests](https://github.com/juglab/napari_denoiseg/workflows/tests/badge.svg)](https://github.com/juglab/napari_denoiseg/actions)
[![codecov](https://codecov.io/gh/juglab/napari_denoiseg/branch/main/graph/badge.svg)](https://codecov.io/gh/juglab/napari_denoiseg)
[![napari hub](https://img.shields.io/endpoint?url=https://api.napari-hub.org/shields/napari-denoiseg)](https://napari-hub.org/plugins/napari-denoiseg)

A napari plugin for self-supervised denoising and segmentation of microscopy images, using the [DenoiSeg] method.

----------------------------------

This [napari] plugin was generated with [Cookiecutter] using [@napari]'s [cookiecutter-napari-plugin] template.

## Installation

You can install `napari-denoiseg` via [pip]:

    pip install napari-denoiseg

To install latest development version :

    pip install git+https://github.com/juglab/napari_denoiseg.git

## Short summary

Napari-denoiseg provides several widgets to make the [DenoiSeg] method available in napari with addition widgets 
to provide quality of life improvements. Fiji Users or people with prior experience with DenoiSeg should see the similarities.

- **Sample Data** to easily try out the widgets provided
- **Training** to train the method with your images and labels, either from the file system or as layers in napari
- **Prediction** to use a trained model from the widget before on new data,  either from the file system or as a layer in napari
- **Patch Creation** as an interactive way to created patches of your ROIs with the right size for training
- **Threshold optimizer** ????


Add example before/after here.

#### What you need

- (optional) a Tensorflow2 compatible GPU to train the model on for better training speed
- Noisy images with pixel-independent noise
- **Some** of these images labeled. Uou can do the labeling inside napari.

## Long summary

The following section describes a possible workflow on how to use the available widgets in this plugin.

Add workflow graphic for that here. Each widget should contain one (or more) gif to show its usage.

### Patch Creation
The patch creation widget is used to create small quadratic/cubic patches in 2D or 3D of your images. 
These patches should include the ROI of your images and can then be used to train the model in the next step more effectively.

GIF here

Set the patch size and then enable the selection. Your cursor now has a rectangle following it along. 
With a simple left-click of your mouse you can select ROIs. After you are down, select a path and click on "Save"
in the widget to write all selected to disk. Changing the patch size clears all previous selections.
If you want to remove certain patches, switch to the "selection" and use the core tools of the napari 
shape layer to delete them.

### DenoiSeg Train
The train widget offers two ways to select input data. You can either choose an image and a label layer that represent 
your data or specify two folders that contain those images. After that, specify the train/test ratio you want to have
your images randomly split in.
After that, you have to specify the axes of your images. Accepted axes are S(ample), T(ime), Z, Y, X, C(channel). Red
color highlighting means that a character is not recognized, orange means that the axes order is not allowed. 
The YX axes are mandatory.

GIF GOES BRRRRR.


Now you need to select if you want to train with 3D patches and set the training parameters to your preferences.
Sensible default values are already preset but might need to be tuned further to match your needs.
You can now start the training and can watch the loss progression in the graph at the bottom.
If you are satisfied with your training result, you can either save the model as a keras h5 file or as a 
bioimage-io model.zip, ready to be uploaded to the [BioImage Model Zoo].

### Threshold Optimizer
@Joran please add a part about the threshold optimizer and how it works.


### DenoiSeg Predict
The prediction widget is used on the data you want denoise and segment. You can either select a certain layer or (lazily)
load from disk, just like before. You need a trained model from the training step before for this to work.
Again, you need to specify if you have 3D images and which axes your data provides.
If you know an optimal threshold already or used the threshold optimized widget from the step before, 
you can set the value now.
Finally, click on "Predict" and the result will be output into a new layer.

GIF AGAIN

## Contributing

Contributions are very welcome. Tests can be run with [tox], please ensure
the coverage at least stays the same before you submit a pull request.

## License

Distributed under the terms of the [BSD-3] license,
"napari-denoiseg" is free and open source software

## Issues

If you encounter any problems, please [file an issue] along with a detailed description and steps to replicate the problem.

[napari]: https://github.com/napari/napari
[Cookiecutter]: https://github.com/audreyr/cookiecutter
[@napari]: https://github.com/napari
[MIT]: http://opensource.org/licenses/MIT
[BSD-3]: http://opensource.org/licenses/BSD-3-Clause
[GNU GPL v3.0]: http://www.gnu.org/licenses/gpl-3.0.txt
[GNU LGPL v3.0]: http://www.gnu.org/licenses/lgpl-3.0.txt
[Apache Software License 2.0]: http://www.apache.org/licenses/LICENSE-2.0
[Mozilla Public License 2.0]: https://www.mozilla.org/media/MPL/2.0/index.txt
[cookiecutter-napari-plugin]: https://github.com/napari/cookiecutter-napari-plugin

[file an issue]: https://github.com/juglab/napari_denoiseg/issues

[napari]: https://github.com/napari/napari
[tox]: https://tox.readthedocs.io/en/latest/
[pip]: https://pypi.org/project/pip/
[PyPI]: https://pypi.org/
[DenoiSeg]: https://github.com/juglab/DenoiSeg
[BioImage Model Zoo]: (https://bioimage.io/)