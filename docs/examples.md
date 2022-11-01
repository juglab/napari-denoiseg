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

In order to reproduce similar the result using the plugin, we then follow these steps:

1. In napari, go to `File / Open sample / napari-denoiseg / Download 2D data (n20 noise)`, after the time necessary to download the data, it will automatically add the BSD68 data set to napari.
2. Confirm that your environment is properly set for GPU training by checking that the GPU indicator (top right) in the plugin displays a greenish GPU label.
3. Set the percentage of training labels to `50%`.
4. In `Training parameters`, set: <br>
`Batch size` = 128 <br>
    > We use different settings here because we have less patches that in the notebook. 
5. You can compare the configuration above to the rest of the `Expert settings` to confirm that the other default values are properly set.
6. Train!

> If your GPU is too small for the training parameters (loading batches in the GPU memory creates out-of-memory errors), then you should decrease the `Batch size` parameter.
> If you reduce the `batch size`, you might want to increase the `N steps`.

## 3D

The [example notebook](https://github.com/juglab/DenoiSeg/blob/main/examples/DenoiSeg_3D/Mouse_Organoid_Cells_CBG_Example.ipynb) 
generates a configuration containing all the parameters used for training and reproducing the results in the DenoiSegConfig call:

In order to reproduce similar the result using the plugin, we then follow these steps:

1. In napari, go to `File / Open sample / napari-denoiseg / Download 3D data (n20 noise)`, after the time necessary to download the data, it will automatically add the BSD68 data set to napari.
2. Confirm that your environment is properly set for GPU training by checking that the GPU indicator (top right) in the plugin displays a greenish GPU label.
3. Set the percentage of training labels to `50%`.
4. In `Training parameters`, set: <br>
`N epochs` = 30 <br>
`N steps` = 150 <br>
`Batch size` = 4 <br>
    > We use different settings here because we have less patches that in the notebook. 
5. You can compare the configuration above to the rest of the `Expert settings` to confirm that the other default values are properly set.
6. Train!

If your GPU is too small for the training parameters (loading batches in the GPU memory creates out-of-memory errors), then you should decrease the `Batch size` parameter.
