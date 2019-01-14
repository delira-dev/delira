[![Build Status](https://travis-ci.com/justusschock/delira.svg?branch=master)](https://travis-ci.com/justusschock/delira) [![Documentation Status](https://readthedocs.org/projects/delira/badge/?version=latest)](https://delira.readthedocs.io/en/latest/?badge=latest) [![codecov](https://codecov.io/gh/justusschock/delira/branch/master/graph/badge.svg)](https://codecov.io/gh/justusschock/delira)

![logo](docs/_static/logo/delira.svg "delira - Deep Learning in Radiology")

# Delira - Deep Learning in Radiology
Authors: [Justus Schock, Christoph Haarburger, Oliver Rippel](AUTHORS.rst)

## Introduction
Delira was developed as a deep learning framework for medical images such as CT or MRI. Currently, it works on arbitrary data (based on [NumPy](http://www.numpy.org/)). 

Based on [PyTorch](https://pytorch.org), [batchgenerators](https://github.com/MIC-DKFZ/batchgenerators) and [trixi](https://github.com/MIC-DKFZ/trixi) it provides a framework for
* Dataset loading
* Dataset sampling
* Augmentation (multi-threaded) including 3D images with any number of channels
* A generic trainer class that implements the training process
* Already implemented [models](delira/models) used in medical image processing and exemplaric implementations of most used models in general (like Resnet)
* Web-based monitoring using [Visdom](https://github.com/facebookresearch/visdom)
* Model save and load functions

Delira supports classification and regression problems as well as generative adversarial networks.

## Installation

### Choose Backend

Currently the only available backend is [PyTorch](https://pytorch.org) (or no backend at all) but we are working on support for [TensorFlow](https://tensorflow.org) as well.  If you want to add another backend, please open an issue (it should not be hard at all) and we will guide you during the process of doing so.

For instructions to install `delira` with a specific backend, please have a look at [the corresponding docs](https://delira.readthedocs.io/en/latest/getting_started.html#installation)

### Installation without a backend (from source)
To install `delira` without a backend (not all functionalities may be work due to a missing backend) you can simply run:
* `pip install git+https://github.com/justusschock/delira.git`

## Getting Started
The best way to learn how to use is to have a look at the [tutorial notebook](notebooks/tutorial_delira.ipynb).
Example implementations for classification problems, segmentation approaches and GANs are also provided in the [notebooks](notebooks) folder.

## Contributing
If you find a bug or have an idea for an improvement, please have a look at our [contribution guideline](CONTRIBUTING.md).
