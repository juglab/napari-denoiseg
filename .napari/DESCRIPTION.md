

![](https://github.com/juglab/DenoiSeg/raw/master/resources/teaser.png)

# Description

This plugin brings the full functionality of the DenoiSeg method as well as several quality of life improvements
directly to napari.

Taken from the abstract of the [DenoiSeg paper]:

Microscopy image analysis often requires the segmentation of objects, 
but training data for this task is typically scarce and hard to obtain. 
Here we propose DenoiSeg, a new method that can be trained end-to-end on 
only a few annotated ground truth segmentations. We achieve this by extending 
Noise2Void, a self-supervised denoising scheme that can be trained on 
noisy images alone, to also predict dense 3-class segmentations. 
The reason for the success of our method is that segmentation can profit 
from denoising, especially when performed jointly within the same network. 
The network becomes a denoising expert by seeing all available raw data, 
while co-learning to segment, even if only a few segmentation labels are available. 
This hypothesis is additionally fueled by our observation that the 
best segmentation results on high quality (very low noise) raw data are obtained 
when moderate amounts of synthetic noise are added. 
This renders the denoising-task non-trivial and unleashes the desired co-learning effect. 
We believe that DenoiSeg offers a viable way to circumvent the tremendous hunger 
for high quality training data and effectively enables few-shot learning of dense segmentations.


![Example GIF hosted on Imgur](https://i.imgur.com/A5phCX4.gif)

Note that GIFs larger than 5MB won't be rendered by GitHub - we will however,
render them on the napari hub.

The other alternative, if you prefer to keep a video, is to use GitHub's video
embedding feature.

1. Push your `DESCRIPTION.md` to GitHub on your repository (this can also be done
as part of a Pull Request)
2. Edit `.napari/DESCRIPTION.md` **on GitHub**.
3. Drag and drop your video into its desired location. It will be uploaded and
hosted on GitHub for you, but will not be placed in your repository.
4. We will take the resolved link to the video and render it on the hub.

Here is an example of an mp4 video embedded this way.

https://user-images.githubusercontent.com/17995243/120088305-6c093380-c132-11eb-822d-620e81eb5f0e.mp4

# Intended Audience & Supported Data

This plugin is intended for usage by image analysts and microbiologists that want to
segment microscopy images but have only little ground truth data available.

The data can be in all by napari loadable formats with two(YX) to five dimensions(STZYXC).
The image axes do not need to be in a specific order, as the user will have to specify that.

The way the method works, both noisy images and some labeled images are necessary.

# Quickstart

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

# Additional Install Steps (uncommon)
We will be providing installation instructions on the hub, which will be sufficient
for the majority of plugins. They will include instructions to pip install, and
to install via napari itself.

Most plugins can be installed out-of-the-box by just specifying the package requirements
over in `setup.cfg`. However, if your plugin has any more complex dependencies, or 
requires any additional preparation before (or after) installation, you should add 
this information here.

# Getting Help

### Contributing

Contributions are very welcome. Tests can be run with [tox], please ensure
the coverage at least stays the same before you submit a pull request.

### Questions?

If you have any questions regarding this plugin, feel free to contact the authors and maintainers 
either on the image.sc forum or the napari zulipchat.

### Issues

If you encounter any problems, please [file an issue] along with a detailed description and steps to replicate the problem.


# How to Cite

```
@inproceedings{BuchholzPrakash2020DenoiSeg,
  title={DenoiSeg: Joint Denoising and Segmentation},
  author={Tim-Oliver Buchholz and Mangal Prakash and Alexander Krull and Florian Jug},
  year={2020}
}
```


[DenoiSeg paper]: https://arxiv.org/abs/2005.02987