.. role:: hidden
    :class: hidden-section

.. currentmodule:: delira.training.callbacks

Callbacks
=========

Callbacks are essential to provide a uniform API for tasks like early stopping
etc.
The PyTorch learning rate schedulers are also implemented as callbacks.
Every callback should ber derived from :class:`AbstractCallback` and must
provide the methods ``at_epoch_begin``
and ``at_epoch_end``.

:hidden:`AbstractCallback`
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: AbstractCallback
    :members:
    :undoc-members:
    :show-inheritance:

:hidden:`EarlyStopping`
~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: EarlyStopping
    :members:
    :undoc-members:
    :show-inheritance:

:hidden:`DefaultPyTorchSchedulerCallback`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: DefaultPyTorchSchedulerCallback
    :members:
    :undoc-members:
    :show-inheritance:

.. currentmodule:: delira.training.callbacks.pytorch_schedulers

:hidden:`CosineAnnealingLRCallback`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: CosineAnnealingLRCallback
    :members:
    :undoc-members:
    :show-inheritance:

:hidden:`ExponentialLRCallback`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: ExponentialLRCallback
    :members:
    :undoc-members:
    :show-inheritance:

:hidden:`LambdaLRCallback`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: LambdaLRCallback
    :members:
    :undoc-members:
    :show-inheritance:

:hidden:`MultiStepLRCallback`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: MultiStepLRCallback
    :members:
    :undoc-members:
    :show-inheritance:

:hidden:`ReduceLROnPlateauCallback`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: ReduceLROnPlateauCallback
    :members:
    :undoc-members:
    :show-inheritance:

:hidden:`StepLRCallback`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: StepLRCallback
    :members:
    :undoc-members:
    :show-inheritance:

.. currentmodule:: delira.training.callbacks

:hidden:`CosineAnnealingLRCallbackPyTorch`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: CosineAnnealingLRCallbackPyTorch
    :members:
    :undoc-members:
    :show-inheritance:

:hidden:`ExponentialLRCallbackPyTorch`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: ExponentialLRCallbackPyTorch
    :members:
    :undoc-members:
    :show-inheritance:

:hidden:`LambdaLRCallbackPyTorch`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: LambdaLRCallbackPyTorch
    :members:
    :undoc-members:
    :show-inheritance:

:hidden:`MultiStepLRCallbackPyTorch`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: MultiStepLRCallbackPyTorch
    :members:
    :undoc-members:
    :show-inheritance:

:hidden:`ReduceLROnPlateauCallbackPyTorch`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: ReduceLROnPlateauCallbackPyTorch
    :members:
    :undoc-members:
    :show-inheritance:

:hidden:`StepLRCallbackPyTorch`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: StepLRCallbackPyTorch
    :members:
    :undoc-members:
    :show-inheritance:

