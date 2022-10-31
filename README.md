# napari-denoiseg

[![License](https://img.shields.io/pypi/l/napari-denoiseg.svg?color=green)](https://github.com/juglab/napari-denoiseg/raw/main/LICENSE)
[![PyPI](https://img.shields.io/pypi/v/napari-denoiseg.svg?color=green)](https://pypi.org/project/napari-denoiseg)
[![Python Version](https://img.shields.io/pypi/pyversions/napari-denoiseg.svg?color=green)](https://python.org)
[![tests](https://github.com/juglab/napari-denoiseg/workflows/build/badge.svg)](https://github.com/juglab/napari-denoiseg/actions)
[![codecov](https://codecov.io/gh/juglab/napari-denoiseg/branch/main/graph/badge.svg)](https://codecov.io/gh/juglab/napari-denoiseg)
[![napari hub](https://img.shields.io/endpoint?url=https://api.napari-hub.org/shields/napari-denoiseg)](https://napari-hub.org/plugins/napari-denoiseg)

A napari plugin performing joint denoising and segmentation of microscopy images using [DenoiSeg](https://github.com/juglab/DenoiSeg).

<img src="https://raw.githubusercontent.com/juglab/napari-denoiseg/master/docs/images/example.png" width="800" />
----------------------------------

## Installation

You can install `napari-denoiseg` via [pip]:
```bash
    pip install napari-denoiseg
```
Or through the [napari-hub](https://napari.org/stable/plugins/find_and_install_plugin.html).


Check out the [documentation](https://juglab.github.io/napari-denoiseg/installation.html) for more detailed installation 
instructions. 


## Quick demo

You can try out a demo by loading the `DenoiSeg Demo prediction` plugin and directly clicking on `Predict`.


<img src="https://raw.githubusercontent.com/juglab/napari-denoiseg/master/docs/images/prediction.gif" width="800" />


## Documentation

Documentation is available on the [project website](https://juglab.github.io/napari-denoiseg/).


## Contributing and feedback

Contributions are very welcome. Tests can be run with [tox], please ensure
the coverage at least stays the same before you submit a pull request. You can also 
help us improve by [filing an issue] along with a detailed description or contact us
through the [image.sc](https://forum.image.sc/) forum (tag @jdeschamps).


## Cite us


Tim-Oliver Buchholz, Mangal Prakash, Alexander Krull and Florian Jug, "[DenoiSeg: Joint Denoising and Segmentation](https://arxiv.org/abs/2005.02987)" _arxiv_ (2020)


## Acknowledgements

This plugin was developed thanks to the support of the Silicon Valley Community Foundation (SCVF) and the 
Chan-Zuckerberg Initiative (CZI) with the napari Plugin Accelerator grant _2021-239867_.


Distributed under the terms of the [BSD-3] license,
"napari-denoiseg" is a free and open source software.

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

[filing an issue]: https://github.com/juglab/napari-denoiseg/issues

[napari]: https://github.com/napari/napari
[tox]: https://tox.readthedocs.io/en/latest/
[pip]: https://pypi.org/project/pip/
[PyPI]: https://pypi.org/
