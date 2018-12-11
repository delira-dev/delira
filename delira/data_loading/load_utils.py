import numpy as np
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
