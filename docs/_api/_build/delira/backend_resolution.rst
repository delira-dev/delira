.. role:: hidden
    :class: hidden-section

.. currentmodule:: delira

Backend Resolution
==================

These functions are used to determine the installed backends and update the
created config file. They also need to be used, to guard backend specific code,
 when writing code with several backends in one file like this:

``if "YOUR_BACKEND" in delira.get_backends():``

:hidden:`get_backends`
~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: get_backends
