# Example pipelines

The plugins come with sample data that can be loaded into napari using `File/Open sample/napari-denoiseg`. As the images 
are downloaded from a remote server, the process can seem idle for a while before eventually loading the images as napari 
layers.

In this section, we describe how to reproduce the results from the DenoiSeg Github repository using the napari plugins.

> **Important note**: if you are using a GPU with little memory (e.g. 4 GB), then most of the shown 
> settings will not work because the batches will probably not fit in memory. Try reducing the batch
> size while increasing the number of steps. This will obviously increase the running time.

## 2D DSB 2018

The [example notebook](https://github.com/juglab/DenoiSeg/blob/3d_example/examples/DenoiSeg_2D/DSB2018_DenoiSeg_Example.ipynb) 
generates a configuration containing all the parameters used for training and reproducing the results in the DenoiSegConfig call:

```bash
config = 
```

The resulting configuration is:

```bash

```

Here we commented some lines with the equivalent parameters in the napari plugin. Parameters that were not specifically 
set in the `DenoiSegConfig` call are set their default and might not need to be set in the napari plugin either.

In order to reproduce the result using the plugin, we then follow these steps:

1. In napari, go to `File / Open sample / napari-denoiseg / Download data (2D)`, after the time necessary to download the data, it will automatically add the BSD68 data set to napari.
2. Confirm that your environment is properly set for GPU training by checking that the GPU indicator (top right) in the plugin displays a greenish GPU label.
3. Select the validation layer in `Val`.
4. In `Training parameters`, set: <br>
`N epochs` = 200 <br>
`N steps` = 400 <br>
`Batch size` = 128 <br>
`Patch XY` = 64 <br>
5. Click on the gear button to open the `Expert settings` and set: <br>
`U-Net kernel size` = 3 <br>
`U-Net residuals` = True (check) <br>
`Split channels` = False (uncheck) <br>
`N2V radius` = 2 <br>
6. You can compare the configuration above to the rest of the `Expert settings` to confirm that the other default values are properly set.
7. Train!

If your GPU is too small for the training parameters (loading batches in the GPU memory creates out-of-memory errors), then you should decrease the `Batch size` parameter.

## 2D FlyWing

The [example notebook](https://github.com/juglab/DenoiSeg/blob/3d_example/examples/DenoiSeg_2D/FlyWing_DenoiSeg_Example.ipynb) 
generates a configuration containing all the parameters used for training and reproducing the results in the DenoiSegConfig call:

## 2D MouseNuclei

The [example notebook](https://github.com/juglab/DenoiSeg/blob/3d_example/examples/DenoiSeg_2D/MouseNuclei_DenoiSeg_Example.ipynb) 
generates a configuration containing all the parameters used for training and reproducing the results in the DenoiSegConfig call:
