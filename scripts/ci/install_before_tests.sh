#!/usr/bin/env bash

pip install -U pip wheel;
pip install -r requirements/base.txt;
pip install slackclient; # install slackclient for all tests (it is not in requirements on purpose)

if [[ "$BACKEND" == "TFEager" ]]; then
    pip install -r requirements/tensorflow.txt
    pip uninstall -y tensorflow-gpu;
    pip install tensorflow==1.14;
elif [[ "$BACKEND" == "TFGraph" ]]; then
    pip install -r requirements/tensorflow.txt
    pip uninstall -y tensorflow-gpu;
    pip install tensorflow==1.14;
elif [[ "$BACKEND" == "Torch" ]]; then
    pip install -r requirements/torch.txt
elif [[ "$BACKEND" == "TorchScript" ]]; then
    pip install -r requirements/torch.txt
elif [[ "$BACKEND" == "Chainer" ]]; then
    pip install -r requirements/chainer.txt
fi

pip install coverage;
pip install codecov;
