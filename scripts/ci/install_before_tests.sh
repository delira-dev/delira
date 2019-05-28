#!/usr/bin/bash

pip install -U pip wheel;
pip install -r requirements.txt;
pip install -r requirements_extra_tf.txt;
pip uninstall -y tensorflow-gpu;
pip install tensorflow==1.13.1;
pip install -r requirements_extra_torch.txt;
pip install coverage;
pip install codecov;
