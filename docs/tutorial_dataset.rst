
# Dataset Guide (incl. Integration to new API) With Delira v0.3.2 a new
dataset API was introduced to allow for more flexibility and add some
features. This notebook shows the difference between the new and the old
API and provides some examples for newly added features.

## Overview Old API The old dataset API was based on the assumption that
the underlying structure of the data can be described as followed: \*
root \* sample1 \* img1 \* img2 \* label \* sample2 \* img1 \* img2 \*
label \* ...

A single sample was constructed from multiple images which are all
located in the same subdirectory. The corresponding signature of the
``AbstractDataset`` was given by
``data_path, load_fn, img_extensions, gt_extensions``. While most
datasets need a ``load_fn`` to load a single sample and a ``data_path``
to the root directory, ``img_extensions``\ and ``gt_exntensions`` were
often unsed. As a consequence a new dataset needed to be created which
initialises the unused variables with arbitrary values.

## Overview New API The new dataset API was refactored to a more general
approach where only a ``data_path`` to the root directory and a
``load_fn`` for a single sample need to be provided. A simple loading
function (``load_fn``) to generate random data independent from the
given path might be realized as below.

.. code:: ipython3

    import numpy as np
    
    
    def load_random_data(path: str) -> dict:
        """Load random data
    
        Parameters
        ----------
        path : str
            path to sample (not used in this example)
    
        Returns
        -------
        dict
            return data inside a dict
        """
        return {
            'data': np.random.rand(3, 512, 512),
            'label': np.random.randint(0, 10),
            'path': path,
        }
    

When used with the provided BaseDatasets, the return value of the load
function is not limited to dictionaries and might be of any type which
can be added to a list with the ``append`` method.

### New Datasets Some basic datasets are already implemented inside
Delira and should be suitable for most cases. The ``BaseCacheDataset``
saves all samples inside the RAM and thus can only be used if everything
fits inside the memory. ´BaseLazyDataset´ loads the individual samples
on time when they are needed, but might lead to slower training due to
the additional loading time.

.. code:: ipython3

    from delira.data_loading import BaseCacheDataset, BaseLazyDataset
    
    
    # because `load_random_data` does not use the path argument, they can have
    # arbitrary values in this example
    paths = list(range(10))
    
    # create case dataset
    cached_set = BaseCacheDataset(paths, load_random_data)
    
    # create lazy dataset
    lazy_set = BaseLazyDataset(paths, load_random_data)
    
    # print cached data
    print(cached_set[0].keys())
    
    # print lazy data
    print(lazy_set[0].keys())
    

In the above example a list of multiple paths is used as the
``data_path``. ``load_fn`` is called for every element inside the
provided list (can be any iterator). If ``data_path`` is a single
string, it is assumed to be the path to the root directory. In this
case, ``load_fn``\ is called for every element inside the root
directory.

Sometimes, a single file/folder contains multiple samples.
``BaseExtendCacheDataset`` uses the ``extend`` function to add elements
to the internal list. Thus it is assumed that ``load_fn`` provides an
iterable object, where eacht item represents a single data sample.

``AbstractDataset`` is now iterable and can be used directly in
combination with for loops.

.. code:: ipython3

    for cs in cached_set:
        print(cs["path"])

## New Utility Function (Integration to new API) The behavior of the old
API can be replicated with the ``LoadSample``,
``LoadSampleLabel``\ functions. ``LoadSample`` assumes that all needed
images and the label (for a single sample) are located in a directory.
Both functions return a dictionary containing the loaded data.
``sample_ext`` maps keys to iterables. Each iterable defines the names
of the images which should be loaded from the directory. ´sample\_fn´ is
used to load the images which are than stacked inside a single array.

.. code:: ipython3

    from delira.data_loading import LoadSample, LoadSampleLabel
    
    
    def load_random_array(path: str):
        """Return random data
    
        Parameters
        ----------
        path : str
            path to image
    
        Returns
        -------
        np.ndarray
            loaded data
        """
        return np.random.rand(128, 128)
    
    
    # define the function to load a single sample from a directory
    load_fn = LoadSample(
        sample_ext={
            # load 3 data channels
            'data': ['red.png', 'green.png', 'blue.png'],
            # load a singel segmentation channel
            'seg': ['seg.png']
        },
        sample_fn=load_random_array,
        # optionally: assign individual keys a datatype
        dtype={"data": "float", "seg": "uint8"},
        # optioanlly: normalize individual samples
        normalize=["data"])
    
    # Note: in general the function should be called with the path of the
    # directory where the imags are located
    sample0 = load_fn(".")
    
    print("data shape: {}".format(sample0["data"].shape))
    print("segmentation shape: {}".format(sample0["seg"].shape))
    print("data type: {}".format(sample0["data"].dtype))
    print("segmentation type: {}".format(sample0["seg"].dtype))
    print("data min value: {}".format(sample0["data"].min()))
    print("data max value: {}".format(sample0["data"].max()))
    

By default the range is normalized to (-1, 1), but ``norm_fn`` can be
changed to achieve other normalization schemes. Some examples are
included in ``delira.data_loading.load_utils``.

``LoadSampleLabel`` takes an additional argument for the label and a
function to load a label. This functions can be used in combination with
the provided BaseDatasets to replicate (and extend) the old API.
