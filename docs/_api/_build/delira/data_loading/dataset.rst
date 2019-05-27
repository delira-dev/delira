.. role:: hidden
    :class: hidden-section

.. currentmodule:: delira.data_loading

Datasets
********

The Dataset the most basic class and implements the loading of your dataset
elements.
You can either load your data in a lazy way e.g. loading them just at the moment
they are needed or you could preload them and cache them.

Datasets can be indexed by integers and return single samples.

To implement custom datasets you should derive the :class:`AbstractDataset`


:hidden:`AbstractDataset`
~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: AbstractDataset
    :members:
    :undoc-members:
    :show-inheritance:

:hidden:`BaseLazyDataset`
~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: BaseLazyDataset
    :members:
    :undoc-members:
    :show-inheritance:

:hidden:`BaseCacheDataset`
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: BaseCacheDataset
    :members:
    :undoc-members:
    :show-inheritance:

:hidden:`BaseExtendCacheDataset`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: BaseExtendCacheDataset
    :members:
    :undoc-members:
    :show-inheritance:

:hidden:`ConcatDataset`
~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: ConcatDataset
    :members:
    :undoc-members:
    :show-inheritance:

:hidden:`BlankDataset`
~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: BlankDataset
    :members:
    :undoc-members:
    :show-inheritance:

:hidden:`Nii3DLazyDataset`
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: Nii3DLazyDataset
    :members:
    :undoc-members:
    :show-inheritance:

:hidden:`Nii3DCacheDataset`
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: Nii3DCacheDataset
    :members:
    :undoc-members:
    :show-inheritance:

:hidden:`TorchvisionClassificationDataset`:
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: TorchvisionClassificationDataset
    :members:
    :undoc-members:
    :show-inheritance:
