.. role:: hidden
    :class: hidden-section

Models
======

``delira`` comes with it's own model-structure tree - with
:class:`AbstractNetwork` at it's root - and integrates
PyTorch Models (:class:`AbstractPyTorchNetwork`) deeply into the model
structure.
Tensorflow Integration is planned.

.. currentmodule:: delira.models

:hidden:`AbstractNetwork`
~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: AbstractNetwork(type)
    :members:
    :undoc-members:
    :show-inheritance:

:hidden:`AbstractPyTorchNetwork`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: AbstractPyTorchNetwork(type)
    :members:
    :undoc-members:
    :show-inheritance:
    
:hidden:`AbstractTfNetwork`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: AbstractTfNetwork
    :members:
    :undoc-members:
    :show-inheritance:


.. toctree::

    Classification <classification>
    Generative Adversarial Networks <gan>
    Segmentation <segmentation>
