import collections
import os

import numpy as np
from skimage.io import imread
from skimage.transform import resize


def norm_range(mode):
    """
    Closure function for range normalization
    Parameters
    ----------
    mode : str
        '-1,1' normalizes data to range [-1, 1], while '0,1'
        normalizes data to range [0, 1]
    Returns
    -------
    callable
        normalization function
    """
    def norm_fn(data):
        """
        Returns the input data normalized to the range
        Parameters
        ----------
        data : np.ndarray
            data which should be normalized
        Returns
        -------
        np.ndarary
            normalized data
        """
        norm = data - data.min()
        norm = norm / norm.max()
        if mode == '-1,1':
            norm = norm - 0.5
            norm = norm * 2
        elif mode == '0,1':
            pass
        else:
            raise ValueError('{mode} not supported.')
        return norm
    return norm_fn


def norm_zero_mean_unit_std(data):
    """
    Return normalized data with mean 0, standard deviation 1
    Parameters
    ----------
    data : np.nadarray
    Returns
    -------
    np.ndarray
        normalized data
    """
    return (data - np.mean(data)) / np.std(data)


class LoadSample:
    """
    Provides a callable to load a single sample from multiple files in a folder
    """

    def __init__(self,
                 sample_ext: dict,
                 sample_fn: collections.abc.Callable,
                 dtype: dict = None, normalize: tuple = (),
                 norm_fn=norm_range('-1,1'),
                 **kwargs):
        """
        Parameters
        ----------
        sample_ext : dict of iterable
            Defines the data _sample_ext. The dict key defines the position of
            the sample inside the returned data dict, while the list defines
            the the files which should be loaded inside the data dict.
        sample_fn : function
            function to load a single sample
        dtype : dict
            defines the data type which should be used for the respective key
        normalize : iterable of hashable
            list of hashable which should be normalized. Can contain
            entire keys of extension (normalizes each element individually)
            or provide the file name which should be normalized
        norm_fn : function
            function to normalize input. Default: normalize range to [-1, 1]
        kwargs :
            variable number of keyword arguments passed to load function
        Examples
        --------
        Simple loading function which returns a dict with `data`
        >>> from delira.data_loading.nii import load_nii
        >>> load_fn = LoadSample({'data:': ['data.nii']}, load_nii)
        Loading function for data (casted to float32 and normalized) and
        segmentation (casted to unit8)
        >>> from delira.data_loading.nii import load_nii
        >>> load_fn = LoadSample({'data:': ['data.nii'], 'seg': ['seg.nii']},
        >>>                      load_nii, dtype={'data': 'float32',
        >>>                                       'seg': 'uint8'},
        >>>                      normalize=('data',))
        """
        if dtype is None:
            dtype = {}
        self._sample_ext = sample_ext
        self._sample_fn = sample_fn
        self._dtype = dtype
        self._normalize = normalize
        self._norm_fn = norm_fn
        self._kwargs = kwargs

    def __call__(self, path) -> dict:
        """
        Load sample from multiple files
        Parameters
        ----------
        path : str
            defines patch to folder which contain the _sample_ext
        Returns
        -------
        dict
            dict with data defines by _sample_ext
        """
        sample_dict = {}
        for key, item in self._sample_ext.items():
            data_list = []
            for f in item:
                data = self._sample_fn(os.path.join(path, f), **self._kwargs)

                # _normalize data if necessary
                if (key in self._normalize) or (f in self._normalize):
                    data = self._norm_fn(data)

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
    def __init__(self,
                 sample_ext: dict,
                 sample_fn: collections.abc.Callable,
                 label_ext: str,
                 label_fn: collections.abc.Callable,
                 dtype: dict = None, normalize: tuple = (),
                 norm_fn=norm_range('-1,1'),
                 sample_kwargs=None, **kwargs):
        """
        Load sample and label from folder
        Parameters
        ----------
        sample_ext : dict of list
            Defines the data _sample_ext. The dict key defines the position of
            the sample inside the returned data dict, while the list defines
            the the files which should be loaded inside the data dict.
            Passed to LoadSample.
        sample_fn : function
            function to load a single sample
            Passed to LoadSample.
        label_ext : str
            extension for label
        label_fn: function
            functions which returns the label inside a dict
        dtype : dict
            defines the data type which should be used for the respective key
        normalize : iterable of hashable
            list of hashable which should be normalized. Can contain
            entire keys of extension (normalizes each element individually)
            or provide the file name which should be normalized
        norm_fn : function
            function to normalize input. Default: normalize range to [-1, 1]
        sample_kwargs :
            additional keyword arguments passed to LoadSample
        kwargs :
            variable number of keyword arguments passed to _label_fn
        See Also
        --------
        :class: `LoadSample`
        """
        if sample_kwargs is None:
            sample_kwargs = {}

        super().__init__(sample_ext=sample_ext, sample_fn=sample_fn,
                         dtype=dtype, normalize=normalize, norm_fn=norm_fn,
                         **sample_kwargs)
        self._label_ext = label_ext
        self._label_fn = label_fn
        self._label_kwargs = kwargs

    def __call__(self, path) -> dict:
        """
        Loads a sample and a label
        Parameters
        ----------
        path : str
        Returns
        -------
        dict
            dict with data and label
        """
        sample_dict = super().__call__(path)
        label_dict = self._label_fn(os.path.join(path, self._label_ext),
                                    **self._label_kwargs)
        sample_dict.update(label_dict)
        return sample_dict
