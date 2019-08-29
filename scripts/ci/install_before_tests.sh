#!/usr/bin/env bash

pip install -U pip wheel;
pip install -r requirements/base.txt;

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
else
    pip install slackclient==1.3.1
fi

pip install coverage;
pip install codecov;
