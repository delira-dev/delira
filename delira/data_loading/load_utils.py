import numpy as np
import os
import typing
import collections
from skimage.io import imread
from skimage.transform import resize


def default_load_fn_2d(img_file, *label_files, img_shape, n_channels=1):
    """
    loading single 2d sample with arbitrary number of samples

    Parameters
    ----------
    img_file : string
        path to image file
    label_files : list of strings
        paths to label files
    img_shape : iterable
        shape of image
    n_channels : int
        number of image channels

    Returns
    -------
    numpy.ndarray
        image
    Any:
        labels

    """
    img = imread(img_file, n_channels == 1)
    labels = [np.loadtxt(_file).reshape(1).astype(np.float32) for _file in
              label_files]
    img = resize(img, img_shape, mode='reflect', anti_aliasing=True)
    img = np.reshape(img, (*img_shape, n_channels))
    img = img.transpose((len(img_shape), *range(len(img_shape))))
    img = img.astype(np.float32)
    # return img, labels

    result_dict = {"data": img}

    for idx, label in enumerate(labels):
        result_dict["label_%d" % idx] = label

    return result_dict


class LoadSample:
    def __init__(self, extensions: typing.Dict[collections.Iterable],
                 sample_fn: collections.Callable, normalize=[], dtype={},
                 **kwargs):
        """
        Provides a callable to load a single sample fom multiple files in a
        folder

        Parameters
        ----------
        extensions: dict of iterable
            Defines the data _extensions. The dict key defines the position of
            the sample inside the returned data dict, while the list defines
            the the files which should be loaded inside the data dict.
        sample_fn: callable
            function to load a single sample
        normalize: list of hashable
            list of hashable which should be normalized. Can contain
            entire keys of extension (normalizes each element individually)
            or provide the file name which should be normalized
        dtype: dict
            defines the data type which should be used for the respective key
        kwargs:
            variable number of keyword arguments passed to load function
        """
        self._extensions = extensions
        self._normalize = normalize
        self._dtype = dtype
        self._sample_fn = sample_fn
        self._kwargs = kwargs

    def __call__(self, path):
        """
        Load sample from multiple files

        Parameters
        ----------
        path: str
            defines patch to folder which contain the _extensions

        Returns
        -------
        dict
            dict with data defines by _extensions
        """
        sample_dict = {}
        for key, item in self._extensions.items():
            data_list = []
            for f in item:
                data = self._sample_fn(os.path.join(path, f), **self._kwargs)

                # _normalize data if necessary
                if (key in self._normalize) or (f in self._normalize):
                    data -= data.min()
                    data = data/data.max()

                # cast data to type
                if key in self._dtype:
                    data = data.astype(self._dtype[key])

                # append data
                data_list.append(data)
            if len(data_list) == 1:
                sample_dict[key] = data_list[0][np.newaxis]
            else:
                sample_dict[key] = np.stack(data_list)
        return sample_dict


class LoadSampleLabel(LoadSample):
    def __init__(self, extensions: typing.Dict[collections.Iterable],
                 sample_fn: collections.Callable,
                 label_ext: collections.Iterable,
                 label_fn: collections.Callable,
                 normalize=[], dtype={}, sample_kwargs={}, **kwargs):
        """
        Load sample and label from folder

        Parameters
        ----------
        extensions: dict of list
            Defines the data _extensions. The dict key defines the position of
            the sample inside the returned data dict, while the list defines
            the the files which should be loaded inside the data dict.
            Passed to LoadSample.
        sample_fn: callable
            function to load a single sample
            Passed to LoadSample.
        normalize: list of hashable
            list of hashable which should be normalized. Can contain
            entire keys of extension (normalizes each element individually)
            or provide the file name which should be normalized
            Passed to LoadSample.
        label_ext: str
            extension for label
        label_fn: function
            functions which returns the label inside a dict
        args:
            variable number of positional arguments passed to LoadSample
        sample_kwargs:
            additional keyword arguments passed to LoadSample
        kwargs:
            variable number of keyword arguments passed to _label_fn

        See Also
        --------
        :class: `LoadSample`
        """
        super().__init__(extensions, sample_fn, normalize, dtype, **sample_kwargs)
        self._label_ext = label_ext
        self._label_fn = label_fn
        self._label_kwargs = kwargs

    def __call__(self, path):
        """
        Loads a sample and a label

        Parameters
        ----------
        path: str

        Returns
        -------
        dict
            dict with data and label
        """
        sample_dict = super(LoadSampleLabel, self).__call__(path)
        label_dict = self._label_fn(os.path.join(path, self._label_ext),
                                    **self._label_kwargs)
        sample_dict.update(label_dict)
        return sample_dict
