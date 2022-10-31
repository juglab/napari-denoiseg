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
conf = DenoiSegConfig(X, unet_kern_size=3, n_channel_out=4, relative_weights = [1.0,1.0,5.0],
                      train_steps_per_epoch=train_steps_per_epoch, train_epochs=10, 
                      batch_norm=True, train_batch_size=train_batch_size, unet_n_first = 32, 
                      unet_n_depth=4, denoiseg_alpha=0.5, train_tensorboard=True)
```

The resulting configuration is:

```bash
{'means': ['13.499731'],
 'stds': ['27.133533'],
 'n_dim': 2,
 'axes': 'YXC',
 'n_channel_in': 1,
 'n_channel_out': 4,
 'train_loss': 'denoiseg',
 'unet_n_depth': 4,
 'relative_weights': [1.0, 1.0, 5.0],
 'unet_kern_size': 3,
 'unet_n_first': 32,
 'unet_last_activation': 'linear',
 'probabilistic': False,
 'unet_residual': False,
 'unet_input_shape': (None, None, 1),
 'train_epochs': 10,
 'train_steps_per_epoch': 237,
 'train_learning_rate': 0.0004,
 'train_batch_size': 128,
 'train_tensorboard': True,
 'train_checkpoint': 'weights_best.h5',
 'train_checkpoint_last': 'weights_last.h5',
 'train_checkpoint_epoch': 'weights_now.h5',
 'train_reduce_lr': {'monitor': 'val_loss', 'factor': 0.5, 'patience': 10},
 'batch_norm': True,
 'n2v_perc_pix': 1.5,
 'n2v_patch_shape': (64, 64),
 'n2v_manipulator': 'uniform_withCP',
 'n2v_neighborhood_radius': 5,
 'denoiseg_alpha': 0.5}
```

In order to reproduce the result using the plugin, we then follow these steps:

1. In napari, go to `File / Open sample / napari-denoiseg / Download 2D data (n20 noise)`, after the time necessary to download the data, it will automatically add the BSD68 data set to napari.
2. Confirm that your environment is properly set for GPU training by checking that the GPU indicator (top right) in the plugin displays a greenish GPU label.
3. Select the validation layer in `Val`.
4. In `Training parameters`, set: <br>
`N epochs` = 15 <br>
`N steps` = 200 <br>
`Batch size` = 128 <br>
`Patch XY` = 64 <br>
    > We use different settings here because we have less patches that in the notebook. 
5. You can compare the configuration above to the rest of the `Expert settings` to confirm that the other default values are properly set.
6. Train!

If your GPU is too small for the training parameters (loading batches in the GPU memory creates out-of-memory errors), then you should decrease the `Batch size` parameter.

## 2D FlyWing

The [example notebook](https://github.com/juglab/DenoiSeg/blob/3d_example/examples/DenoiSeg_2D/FlyWing_DenoiSeg_Example.ipynb) 
generates a configuration containing all the parameters used for training and reproducing the results in the DenoiSegConfig call:

## 2D MouseNuclei

The [example notebook](https://github.com/juglab/DenoiSeg/blob/3d_example/examples/DenoiSeg_2D/MouseNuclei_DenoiSeg_Example.ipynb) 
generates a configuration containing all the parameters used for training and reproducing the results in the DenoiSegConfig call:
