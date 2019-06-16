#!/usr/bin/env bash

pip install -U pip wheel;
pip install -r requirements.txt;

if [[ "$BACKEND" == "TF" ]]; then
    pip install -r requirements/tensorflow.txt
    pip uninstall -y tensorflow-gpu;
    pip install tensorflow==1.13.1;
elif [[ "$BACKEND" == "Torch" ]]; then
    pip install -r requirements/torch.txt
elif [[ "$BACKEND" == "Chainer" ]]; then
    pip install -r requirements/chainer.txt
fi

pip install coverage;
pip install codecov;
