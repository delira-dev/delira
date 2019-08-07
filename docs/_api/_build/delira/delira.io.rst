.. role:: hidden
    :class: hidden-section

IO
==

.. currentmodule:: delira.io

if "CHAINER" in get_backends():
    from delira.io.chainer import save_checkpoint as chainer_save_checkpoint
    from delira.io.chainer import load_checkpoint as chainer_load_checkpoint

if "SKLEARN" in get_backends():
    from delira.io.sklearn import load_checkpoint as sklearn_load_checkpoint
    from delira.io.sklearn import save_checkpoint as sklearn_save_checkpoint


:hidden:`torch_load_checkpoint`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: torch_load_checkpoint

:hidden:`torch_save_checkpoint`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: torch_save_checkpoint

:hidden:`torchscript_load_checkpoint`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: torchscript_load_checkpoint

:hidden:`torchscript_save_checkpoint`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: torchscript_save_checkpoint

:hidden:`tf_load_checkpoint`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: tf_load_checkpoint

:hidden:`tf_save_checkpoint`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: tf_save_checkpoint

:hidden:`tf_eager_load_checkpoint`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: tf_eager_load_checkpoint

:hidden:`tf_eager_save_checkpoint`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: tf_eager_save_checkpoint

:hidden:`chainer_load_checkpoint`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: chainer_load_checkpoint

:hidden:`chainer_save_checkpoint`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: chainer_save_checkpoint

:hidden:`sklearn_load_checkpoint`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: sklearn_load_checkpoint

:hidden:`sklearn_save_checkpoint`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: sklearn_save_checkpoint
