

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

Try to make the data description as explicit as possible, so that users know the
format your plugin expects. This applies both to reader plugins reading file formats
and to function/dock widget plugins accepting layers and/or layer data.
For example, if you know your plugin only works with 3D integer data in "tyx" order,
make sure to mention this.

If you know of researchers, groups or labs using your plugin, or if it has been cited
anywhere, feel free to also include this information here.

# Quickstart

This section should go through step-by-step examples of how your plugin should be used.
Where your plugin provides multiple dock widgets or functions, you should split these
out into separate subsections for easy browsing. Include screenshots and videos
wherever possible to elucidate your descriptions. 

Ideally, this section should start with minimal examples for those who just want a
quick overview of the plugin's functionality, but you should definitely link out to
more complex and in-depth tutorials highlighting any intricacies of your plugin, and
more detailed documentation if you have it.

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