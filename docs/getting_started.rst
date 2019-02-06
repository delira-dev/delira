Getting started
===============

Installation
------------

Backends
~~~~~~~~~~~

Before installing ``delira``, you have to choose a suitable backend.
``delira`` handles backends as optional dependencies and tries to escape all uses of a not-installed backend.

The currently supported backends are:

* `torch <https://pytorch.org>`_ (recommended, since it is the most tested backend): Suffix ``torch``

  .. note::
    ``delira`` supports mixed-precision training via `apex <https://github.com/NVIDIA/apex>`_, but ``apex`` must be installed separately
    
* `tf <https://tensorflow.org>`_ (very experimental): Suffix ``tf``

  .. note::
    the ``tf`` backend is still very experimental and may be unstable.

* None: No Suffix

* All (installs all registered backends and their dependencies; not recommended, since this will install many large packages): Suffix ``full``

.. note::
  Depending on the backend, some functionalities may not be available for you. If you want to ensure, you can use each functionality, please use the ``full`` option, since it installs all backends
  
.. note:: 
  Currently the only other planned backend is TensorFlow (which is coming soon). If you want to add a backend like `CNTK <https://www.microsoft.com/en-us/cognitive-toolkit/>`_, `Chainer <https://chainer.org/>`_, `MXNET <https://mxnet.apache.org/>`_ or something similar, please open an issue for that and we will guide you during that process (don't worry, it is not much effort at all).

======== ============================= ============================================================================================= ======================================================================================================================
Backend  Binary Installation           Source Installation                                                                           Notes
======== ============================= ============================================================================================= ======================================================================================================================
None     ``pip install delira``        ``pip install git+https://github.com/justusschock/delira.git``                                Training not possible if backend is not installed separately
`torch`_ ``pip install delira[torch]`` ``git clone https://github.com/justusschock/delira.git && cd delira && pip install .[torch]`` ``delira`` with ``torch`` backend supports mixed-precision training via `NVIDIA/apex`_ (must be installed separately).
`tf`_               --                 ``git clone https://github.com/justusschock/delira.git && cd delira && pip install .[tf]``    The tensorflow backend is still very experimental and lacks some `features`_
Full     ``pip install delira[full]``  ``git clone https://github.com/justusschock/delira.git && cd delira && pip install .[full]``  All backends are getting installed.
======== ============================= ============================================================================================= ======================================================================================================================

.. _torch: https://pytorch.org
.. _NVIDIA/apex: https://github.com/NVIDIA/apex.git
.. _tf: https://tensorflow.org
.. _features: https://github.com/justusschock/delira/issues/47
