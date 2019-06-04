#!/usr/bin/env bash

pip install -U pip wheel;
pip install -r requirements.txt;

if [[ "$BACKEND" == "TF" ]]; then
    pip install -r requirements_extra_tf.txt;
    pip uninstall -y tensorflow-gpu;
    pip install tensorflow==1.13.1;
elif [[ "$BACKEND" == "Torch" ]]; then
    pip install -r requirements_extra_torch.txt;
elif [[ "$BACKEND" == "Docs" ]]; then
    pip install -r docs/requirements.txt;
fi

pip install coverage;
pip install codecov;
