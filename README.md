[![Build Status](https://travis-ci.com/justusschock/delira.svg?branch=master)](https://travis-ci.com/justusschock/delira)

![logo](docs/_static/logo/delira.svg "delira - Deep Learning in Radiology")

# Delira - Deep Learning in Radiology
Authors: [Justus Schock, Christoph Haarburger, Oliver Rippel](AUTHORS.rst)

## Introduction
Delira is deep learning framework for medical images such as CT or MRI. Based on [PyTorch](https://pytorch.org) and [batchgenerators](https://github.com/MIC-DKFZ/batchgenerators) it provides a framework for
* Dataset loading
* Dataset sampling
* Augmentation (multi-threaded) including 3D images with any number of channels
* A generic trainer class that implements the training process
* Web-based monitoring using [Visdom](https://github.com/facebookresearch/visdom)
* Model save and load functions

Delira supports classification and regression problems as well as generative adversarial networks.

## Installation
* `pip install git+https://github.com/justusschock/delira.git`

## Getting Started
The best way to learn how to use is to have a look at the [tutorial notebook](https://github.com/justusschock/delira/blob/master/notebooks/tutorial_delira.ipynb).
Example implementations for classification problems and GANs are also provided in the [notebooks](https://github.com/justusschock/delira/blob/master/notebooks) folder.

## Contributing
If you find a bug or have an idea for an improvement, please have a look at our [contribution guideline](CONTRIBUTING.md).
