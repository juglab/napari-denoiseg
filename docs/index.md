# napari-denoiseg

`napari-denoiseg` brings [DenoiSeg](https://github.com/juglab/DenoiSeg) to the fantastic world of napari. DenoiSeg is 
an algorithm allowing the joint denoising and segmentation of microscopy data using little ground-truth annotation. 
DenoiSeg is an offshoot of [Noise2Void](https://github.com/juglab/n2v).

This set of plugins can train, retrain and predict on images from napari or from the disk. It conveniently allows saving 
the models for later use and is compatible with [Bioimage.io](https://bioimage.io/#/). 


<img src="https://raw.githubusercontent.com/juglab/napari-denoiseg/master/docs/images/prediction.gif" width="800" />

# Documentation

1. [Installation](installation.md)
2. [Documentation](documentation.md)
3. [Examples](examples.md)
4. [Troubleshooting](faq.md)

# Report issues and errors

Help us improve the plugin by submitting [issues to the Github repository](https://github.com/juglab/napari-denoiseg/issues) 
or tagging @jdeschamps on [image.sc](https://forum.image.sc/). 

# Cite us

Tim-Oliver Buchholz, Mangal Prakash, Alexander Krull and Florian Jug, "[DenoiSeg: Joint Denoising and Segmentation](https://arxiv.org/abs/2005.02987)" _arxiv_ (2020)


# Acknowledgements

This plugin was developed thanks to the support of the Silicon Valley Community Foundation (SCVF) and the 
Chan-Zuckerberg Initiative (CZI) with the napari Plugin Accelerator grant _2021-239867_.


Distributed under the terms of the [BSD-3](http://opensource.org/licenses/BSD-3-Clause) license,
"napari-denoiseg" is a free and open source software.