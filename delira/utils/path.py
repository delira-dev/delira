import os


def subdirs(d):
    """For a given directory, return a list of all subdirectories (full paths)

    Parameters
    ----------
    d : string
        given root directory

    Returns
    -------
    list
        list of strings of all subdirectories
    """
    return sorted([os.path.join(d, name) for name in os.listdir(d)
                   if os.path.isdir(os.path.join(d, name))])
