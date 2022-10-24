
# Installation

## Create a conda environment

If you do not have conda, we recommend installing [miniconda](https://docs.conda.io/en/latest/miniconda.html) or [Anaconda](https://www.anaconda.com/).

Then, in your command line tool:

1. Create a conda environment
    
    ```bash
    conda create -n 'napari-denoiseg' python=3.9
    conda activate napari-denoiseg
    ```
    
2. Follow the [TensorFlow installation step-by-step](https://www.tensorflow.org/install/pip#linux_1) for your 
operating system.
3. Install `napari`:
    ```bash
    pip install "napari[all]==0.4.15"
    ```


## Install napari-denoiseg

### Install napari-denoiseg through the napari-hub

Check-out the instructions on [installing plugins via the napari hub](https://napari.org/stable/plugins/find_and_install_plugin.html). 
This step is performed after [starting napari](#start-napari-denoiseg).

### Install napari-denoiseg via pip

Within the previously installed conda environment, type:

```bash
pip install mapari-denoiseg
```

### Install napari-denoiseg from source

Clone the repository:
```bash
git clone https://github.com/juglab/napari-denoiseg.git
```

Navigate to the newly created folder:
```bash
cd napari-denoiseg
```

Within the previously installed conda environment, type:

```bash
pip install -e .
```

# Start napari-denoiseg

1. Using the terminal with the `napari-denoiseg` environment active, start napari:
    
    ```bash
    napari
    ```
    
2. Load one of the napari-denoiseg plugin.