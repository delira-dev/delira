.. role:: hidden
    :class: hidden-section

.. currentmodule:: delira.training


NetworkTrainer
==============
The network trainer implements the actual training routine and can be subclassed
 for special routines.
Subclassing your trainer also means you have to subclass your experiment (to
use the trainer).

:hidden:`AbstractNetworkTrainer`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: AbstractNetworkTrainer
    :members:
    :undoc-members:
    :show-inheritance:

:hidden:`PyTorchNetworkTrainer`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: PyTorchNetworkTrainer
    :members:
    :undoc-members:
    :show-inheritance:

:hidden:`TfNetworkTrainer`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: TfNetworkTrainer
    :members:
    :undoc-members:
    :show-inheritance:
