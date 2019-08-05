Getting started
===============

Backends
--------

Before installing ``delira``, you have to choose a suitable backend.
``delira`` handles backends as optional dependencies and tries to escape all uses of a not-installed backend.

The currently supported backends are:

* `torch <https://pytorch.org>`_ (recommended, since it is the most tested backend): Suffix ``torch``

  .. note::
    ``delira`` supports mixed-precision training via `apex <https://github.com/NVIDIA/apex>`_, but ``apex`` must be installed separately
   
* `torchscript <https://pytorch.org/docs/stable/jit.html>`_ : Suffix ``torchscript``

  .. note::
    ``delira`` with ``torchscript`` backend dies currently not support Multi-GPU training.
    
* `tensorflow eager execution <https://tensorflow.org>`_: Suffix ``tensorflow``

  .. note::
    ``delira`` with ``tensorflow eager`` backend dies currently not support Multi-GPU training.

* `tensorflow graph mode <https://tensorflow.org>`_: Suffix ``tensorflow``

  .. note::
    ``delira`` with ``tensorflow graph`` backend dies currently not support Multi-GPU training.

* `chainer <https://chainer.org>`_: Suffix ``chainer``

* `scikit-learn <https://scikit-learn.org/stable/>`_: No Suffix

* None: No Suffix

* All (installs all registered backends and their dependencies; not recommended, since this will install many large packages): Suffix ``full``

.. note::
  Depending on the backend, some functionalities may not be available for you. If you want to ensure, you can use each functionality, please use the ``full`` option, since it installs all backends
  
.. note:: 
  If you want to add a backend like `CNTK <https://www.microsoft.com/en-us/cognitive-toolkit/>`_, `MXNET <https://mxnet.apache.org/>`_ or something similar, please open an issue for that and we will guide you during that process (don't worry, it is not much effort at all).

Installation
------------

=================== =================================== ================================================================================================= ======================================================================================================================
Backend             Binary Installation                 Source Installation                                                                               Notes
=================== =================================== ================================================================================================= ======================================================================================================================
None                ``pip install delira``              ``pip install git+https://github.com/delira-dev/delira.git``                                      Training not possible if backend is not installed separately
`torch`_            ``pip install delira[torch]``       ``git clone https://github.com/delira-dev/delira.git && cd delira && pip install .[torch]``       ``delira`` with ``torch`` backend supports mixed-precision training via `NVIDIA/apex`_ (must be installed separately).
`torchscript`_      ``pip install delira[torchscript]`` ``git clone https://github.com/delira-dev/delira.git && cd delira && pip install .[torchscript]`` The ``torchscript`` backend currently supports only single-GPU-training
`tensorflow eager`_ ``pip install delira[tensorflow]``  ``git clone https://github.com/delira-dev/delira.git && cd delira && pip install .[tensorflow]``  the ``tensorflow`` backend is still very experimental and lacks some `features`_
`tensorflow graph`_ ``pip install delira[tensorflow]``  ``git clone https://github.com/delira-dev/delira.git && cd delira && pip install .[tensorflow]``  the ``tensorflow`` backend is still very experimental and lacks some `features`_
`scikit-learn`_     ``pip install delira``              ``pip install git+https://github.com/delira-dev/delira.git``                                      /
`chainer`_          ``pip install delira[chainer]``     ``git clone https://github.com/delira-dev/delira.git && cd delira && pip install .[chainer]``     /
Full                ``pip install delira[full]``        ``git clone https://github.com/delira-dev/delira.git && cd delira && pip install .[full]``        All backends will be installed
=================== =================================== ================================================================================================= ======================================================================================================================

.. _torch: https://pytorch.org
.. _NVIDIA/apex: https://github.com/NVIDIA/apex.git
.. _torchscript: https://pytorch.org/docs/stable/jit.html
.. _tensorflow eager: https://www.tensorflow.org/
.. _features: https://github.com/delira-dev/delira/issues/47
.. _tensorflow graph: https://www.tensorflow.org/
.. _scikit-learn: https://scikit-learn.org/stable/
.. _chainer: https://chainer.org/