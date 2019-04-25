import logging
import SimpleITK as sitk
import numpy as np
import json
import os
from abc import abstractmethod
from delira.utils.decorators import make_deprecated
logger = logging.getLogger(__name__)


def load_nii(path):
    """
    Loads a single nii file
    Parameters
    ----------
    path: str
        path to nii file which should be loaded

    Returns
    -------
    np.ndarray
        numpy array containing the loaded data
    """
    return sitk.GetArrayFromImage(sitk.ReadImage(path))


@make_deprecated('LoadSample function can be used this replicate the behavior.')
def load_sample_nii(files, label_load_cls):
    """
    Load sample from multiple ITK files

    Parameters
    ----------
    files : dict with keys `img` and `label`
        filenames of nifti files and label file
    label_load_cls : class
        function to be used for label parsing

    Returns
    -------
    dict
        sample: dict with keys `data` and `label` containing images and label

    Raises
    ------
    AssertionError
        if `img.max()` is greater than 511 or smaller than 1

    """
    img_list = []
    for f in files['img']:
        img = sitk.GetArrayFromImage(sitk.ReadImage(f))
        img = img.astype(np.float32)
        assert img.max() <= 511
        assert img.max() > 1
        img = img/511
        img_list.append(img)
    label_gen = label_load_cls(files['label'])
    label = label_gen.get_labels()
    sample = {"data": np.stack(img_list), "label": label}
    if 'mask' in list(files.keys()):
        mask = sitk.GetArrayFromImage(sitk.ReadImage(files['mask']))
        mask = mask.astype(np.int64)
        sample['mask'] = mask
    return sample


@make_deprecated("Labels can now be provided by a function which returns "
                 "a dictionary.")
class BaseLabelGenerator(object):
    """
    Base Class to load labels from json files

    """
    def __init__(self, fpath):
        """

        Parameters
        ----------
        fpath : str
            filepath to json file

        Raises
        ------
        AssertionError
            `fpath` does not end with 'json'

        """
        assert fpath.endswith('json')
        self.fpath = fpath
        self.data = self._load()

    def _load(self):
        """
        Private Helper function to load the file

        Returns
        -------
        Any
            loaded values from file

        """
        with open(os.path.join(self.fpath), 'r') as f:
            label = json.load(f)
        return label

    @abstractmethod
    def get_labels(self):
        """
        Abstractmethod to get labels from class

        Raises
        ------
        NotImplementedError
            if not overwritten in subclass

        """
        raise NotImplementedError()
