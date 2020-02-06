[<img src="https://img.shields.io/badge/chat-slack%20channel-75BBC4.svg">](https://join.slack.com/t/deliradev/shared_invite/enQtNjI1MjA4MjQzMzQ2LTUzNTQ0MjQyNjJjNzgyODczY2Y1YjYxNjA3ZmQ0MGFhODhkYzQ4M2RjMGM1YWM3YWU5MDM0ZjdiNTQ4MmQ0ZDk)
[![PyPI version](https://badge.fury.io/py/delira.svg)](https://badge.fury.io/py/delira) [![Build Status](https://travis-ci.com/delira-dev/delira.svg?branch=master)](https://travis-ci.com/delira-dev/delira) [![Documentation Status](https://readthedocs.org/projects/delira/badge/?version=master)](https://delira.readthedocs.io/en/master/?badge=master) [![codecov](https://codecov.io/gh/justusschock/delira/branch/master/graph/badge.svg)](https://codecov.io/gh/delira-dev/delira)
[![DOI](http://joss.theoj.org/papers/10.21105/joss.01488/status.svg)](https://doi.org/10.21105/joss.01488)

![logo](docs/_static/logo/delira.svg "delira - A Backend Agnostic High Level Deep Learning Library")

# delira - A Backend Agnostic High Level Deep Learning Library
Authors: [Justus Schock, Michael Baumgartner, Oliver Rippel, Christoph Haarburger](AUTHORS.rst)

Copyright (C) 2020 by RWTH Aachen University                      
http://www.rwth-aachen.de                                             
                                                                         
License:                                                                                                                                       
This software is dual-licensed under:                                 
• Commercial license (please contact: lfb@lfb.rwth-aachen.de)         
• AGPL (GNU Affero General Public License) open source license        

## Introduction
`delira` is designed to work as a backend agnostic high level deep learning library. You can choose among several computation [backends](#choose-backend).
It allows you to compare different models written for different backends without rewriting them.

For this case, `delira` couples the entire training and prediction logic in backend-agnostic modules to achieve identical behavior for training in all backends.

`delira` is designed in a very modular way so that almost everything is easily exchangeable or customizable.

A (non-comprehensive) list of the features included in `delira`:
* Dataset loading
* Dataset sampling
* Augmentation (multi-threaded) including 3D images with any number of channels (based on [`batchgenerators`](https://github.com/MIC-DKFZ/batchgenerators))
* A generic trainer class that implements the training process for all [backends](#choose-backend)
* Training monitoring using [Visdom](https://github.com/facebookresearch/visdom) or [Tensorboard](https://www.tensorflow.org/guide/summaries_and_tensorboard)
* Model save and load functions
* Already impelemented Datasets
* Many operations and utilities for medical imaging

## What about the name?
`delira` started as a library to enable deep learning research and fast prototyping in medical imaging (especially in radiology). 
That's also where the name comes from: `delira` was an acronym for **DE**ep **L**earning **I**n **RA**diology*. 
To adapt many other use cases we changed the framework's focus quite a bit, although we are still having many medical-related utilities 
and are working on constantly factoring them out.


## Installation

### Choose Backend

You may choose a backend from the list below. If your desired backend is not listed and you want to add it, please open an issue (it should not be hard at all) and we will guide you during the process of doing so.


| Backend                                                   | Binary Installation               | Source Installation                                                                               | Notes                                                                                                                                                 |
|-----------------------------------------------------------|-----------------------------------|---------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------|
| None                                                      | `pip install delira`              | `pip install git+https://github.com/delira-dev/delira.git`                                      | Training not possible if backend is not installed separately                                                                                          |
| [`torch`](https://pytorch.org)                            | `pip install delira[torch]`       | `git clone https://github.com/delira-dev/delira.git && cd delira && pip install .[torch]`       | `delira` with `torch` backend supports mixed-precision training via [NVIDIA/apex](https://github.com/NVIDIA/apex.git) (must be installed separately). |
| [`torchscript`](https://pytorch.org/docs/stable/jit.html) | `pip install delira[torchscript]` | `git clone https://github.com/delira-dev/delira.git && cd delira && pip install .[torchscript]` | The `torchscript` backend currently supports only single-GPU-training                                                                                 |
| [`tensorflow eager`](https://www.tensorflow.org/)         | `pip install delira[tensorflow]`  | `git clone https://github.com/delira-dev/delira.git && cd delira && pip install .[tensorflow]`  | the `tensorflow` backend is still very experimental and lacks some [features](https://github.com/delira-dev/delira/issues/47)                       |
| [`tensorflow graph`](https://www.tensorflow.org/)         | `pip install delira[tensorflow]`  | `git clone https://github.com/delira-dev/delira.git && cd delira && pip install .[tensorflow]`  | the `tensorflow` backend is still very experimental and lacks some [features](https://github.com/delira-dev/delira/issues/47)                       |
| [`scikit-learn`](https://scikit-learn.org/stable/)        | `pip install delira`              | `pip install git+https://github.com/delira-dev/delira.git`                                      | /                                                                                                                                                     |
| [`chainer`](https://chainer.org/)                         | `pip install delira[chainer]`     | `git clone https://github.com/delira-dev/delira.git && cd delira && pip install .[chainer]`     | /
| Full                                                      | `pip install delira[full]`        | `git clone https://github.com/delira-dev/delira.git && cd delira && pip install .[full]`        | All backends will be installed.                                                                                                                       |

### Docker
The easiest way to use `delira` is via docker (with the [nvidia-runtime](https://github.com/NVIDIA/nvidia-docker) for GPU-support) and using the [Dockerfile](docker/Dockerfile) or the [prebuild-images](https://cloud.docker.com/u/justusschock/repository/docker/justusschock/delira).

### Chat
We have a [community chat on slack](https://deliradev.slack.com). If you need an invitation, just follow [this link](https://join.slack.com/t/deliradev/shared_invite/enQtNjI1MjA4MjQzMzQ2LTUzNTQ0MjQyNjJjNzgyODczY2Y1YjYxNjA3ZmQ0MGFhODhkYzQ4M2RjMGM1YWM3YWU5MDM0ZjdiNTQ4MmQ0ZDk).

## Getting Started
The best way to learn how to use is to have a look at the [tutorial notebook](notebooks/tutorial_delira.ipynb).
Example implementations for classification problems, segmentation approaches and GANs are also provided in the [notebooks](notebooks) folder.

## Documentation
The docs are hosted on [ReadTheDocs/Delira](https://delira.rtfd.io).
The documentation of the latest master branch can always be found at the project's [github page](https://delira-dev.github.io/delira/).

## Contributing
If you find a bug or have an idea for an improvement, please have a look at our [contribution guideline](CONTRIBUTING.md).
