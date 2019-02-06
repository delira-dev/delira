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
    
:hidden:`ConcatDataset`
~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: ConcatDataset
    :members:
    :undoc-members:
    :show-inheritance:
