
# Installation

## Windows/Linux

If you do not have conda, we recommend installing [miniconda](https://docs.conda.io/en/latest/miniconda.html) or [Anaconda](https://www.anaconda.com/).

1. Then, in your command line tool create a conda environment 
   ```bash
       conda create -n napari-denoiseg python=3.9
       conda activate napari-denoiseg
   ```
2. Follow the [TensorFlow installation step-by-step](https://www.tensorflow.org/install/pip#linux_1) for your 
operating system.
3. Install `napari`:
   ```bash
      pip install "napari[all]==0.4.15"
   ```
4. Install `napari-denoiseg`:
   ```bash
      pip install napari-denoiseg
   ```
   
## macOS

> Note: These instructions are for GPU support. Apple's tensorflow-metal is only officially supported for macOS 12 and 
> higher. For CPU, you can try the Follow the [TensorFlow instructions](https://www.tensorflow.org/install/pip#macos_1) 

1. Set up env with napari and pyqt5
   ```bash
      conda create -n napari-denoiseg -c conda-forge python=3.9 napari pyqt5 imagecodecs
   ```
2. Install tensorflow following [Apple's instructions](https://developer.apple.com/metal/tensorflow-plugin/)
3. Install napari-denoiseg
   ```bash
      pip install napari-denoiseg
   ```


# Start napari-denoiseg

1. Using the terminal with the `napari-denoiseg` environment active, start napari:
    
    ```bash
    napari
    ```
    
2. Load one of the napari-denoiseg plugin.