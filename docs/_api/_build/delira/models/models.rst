.. role:: hidden
    :class: hidden-section

Models
======

``delira`` comes with it's own model-structure tree - with
:class:`AbstractNetwork` at it's root - and integrates
several backends deeply into it's structure.

.. currentmodule:: delira.models

:hidden:`AbstractNetwork`
~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: AbstractNetwork(type)
    :members:
    :undoc-members:
    :show-inheritance:

Backends
--------

.. toctree::

    Chainer <chainer>
    SciKit-Learn <sklearn>
    TensorFLow Eager Execution <tfeager>
    TensorFlow Graph Execution <tfgraph>
    PyTorch <torch>
    TorchScript <torchscript>
