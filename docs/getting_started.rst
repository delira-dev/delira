Getting started
===============

Installation
------------

Backends
~~~~~~~~~~~

Before installing ``delira``, you have to choose a suitable backend.
``delira`` handles backends as optional dependencies and tries to escape all uses of a not-installed backend.

The currently supported backends are:

* `torch<https://pytorch.org>` (recommended, since it is the most tested backend): Suuffix `torch`
* None: No Suffix

.. note::
  Depending on the backend, some functionalities may not be available for you. If you want to ensure, you can use each functionality, please use the ``full`` option, since it installs all backends
  
.. note:: 
  Currently the only other planned backend is TensorFlow (which is coming soon). If you want to add a backend like `CNTK<https://www.microsoft.com/en-us/cognitive-toolkit/>`, `Chainer<https://chainer.org/>`, `MXNET<https://mxnet.apache.org/>` or something similar, please open an issue for that and we will guide you during that process (don't worry, it is not much effort at all).

From Source
~~~~~~~~~~~
To install``delira`` you can simply run

* ``pip install git+https://github.com/justusschock/delira.git[suffix]``
 by replacing ``[suffix]`` with the suffix for your backend, i.e. installing ``delira`` with ``torch`` backend would become ``pip install git+https://github.com/justusschock/delira.git[torch]`` and installing without a backend at all would become ``pip install git+https://github.com/justusschock/delira.git``
