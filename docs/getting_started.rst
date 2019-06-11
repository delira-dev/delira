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
    
* `tf <https://tensorflow.org>`_ (very experimental): Suffix ``tensorflow``

  .. note::
    the ``tensorflow`` backend is still very experimental and may be unstable.

* None: No Suffix

* All (installs all registered backends and their dependencies; not recommended, since this will install many large packages): Suffix ``full``

.. note::
  Depending on the backend, some functionalities may not be available for you. If you want to ensure, you can use each functionality, please use the ``full`` option, since it installs all backends
  
.. note:: 
  If you want to add a backend like `CNTK <https://www.microsoft.com/en-us/cognitive-toolkit/>`_, `Chainer <https://chainer.org/>`_, `MXNET <https://mxnet.apache.org/>`_ or something similar, please open an issue for that and we will guide you during that process (don't worry, it is not much effort at all).

Installation
------------

+---------+-----+--------------------+---------------------------------+
| Backend | Bin | Source             | Notes                           |
|         | ary | Installation       |                                 |
|         | Ins |                    |                                 |
|         | tal |                    |                                 |
|         | lat |                    |                                 |
|         | ion |                    |                                 |
+=========+=====+====================+=================================+
| None    | ``p | ``pip install git+ | Training not possible if        |
|         | ip  | https://github.com | backend is not installed        |
|         | ins | /justusschock/deli | separately                      |
|         | tal | ra.git``           |                                 |
|         | l d |                    |                                 |
|         | eli |                    |                                 |
|         | ra` |                    |                                 |
|         | `   |                    |                                 |
+---------+-----+--------------------+---------------------------------+
| `torch` | ``p | ``git clone https: | ``delira`` with ``torch``       |
| _       | ip  | //github.com/justu | backend supports                |
|         | ins | sschock/delira.git | mixed-precision training via    |
|         | tal |  && cd delira && p | `NVIDIA/apex`_ (must be         |
|         | l d | ip install .[torch | installed separately).          |
|         | eli | ]``                |                                 |
|         | ra[ |                    |                                 |
|         | tor |                    |                                 |
|         | ch] |                    |                                 |
|         | ``  |                    |                                 |
+---------+-----+--------------------+---------------------------------+
| `tensor | ``p | ``git clone https: | the ``tensorflow`` backend is   |
| flow`_  | ip  | //github.com/justu | still very experimental and     |
|         | ins | sschock/delira.git | lacks some `features`_          |
|         | tal |  && cd delira && p |                                 |
|         | l d | ip install .[tenso |                                 |
|         | eli | rflow]``           |                                 |
|         | ra[ |                    |                                 |
|         | ten |                    |                                 |
|         | sor |                    |                                 |
|         | flo |                    |                                 |
|         | w]` |                    |                                 |
|         | `   |                    |                                 |
+---------+-----+--------------------+---------------------------------+
| Full    | ``p | ``git clone https: | All backends will be installed. |
|         | ip  | //github.com/justu |                                 |
|         | ins | sschock/delira.git |                                 |
|         | tal |  && cd delira && p |                                 |
|         | l d | ip install .[full] |                                 |
|         | eli | ``                 |                                 |
|         | ra[ |                    |                                 |
|         | ful |                    |                                 |
|         | l]` |                    |                                 |
|         | `   |                    |                                 |
+---------+-----+--------------------+---------------------------------+

.. _torch: https://pytorch.org
.. _NVIDIA/apex: https://github.com/NVIDIA/apex.git
.. _tensorflow: https://www.tensorflow.org/
.. _features: https://github.com/justusschock/delira/issues/47