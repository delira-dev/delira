import chainer
import zipfile
import os
import json


def save_checkpoint(file, model=None, optimizers=None, epoch=None):
    """
    Saves the given checkpoint

    Parameters
    ----------
    file : str
        string containing the path, the state should be saved to
    model : :class:`AbstractChainerNetwork`
    optimizers : dict
        dictionary containing all optimizers
    epoch : int
        the current epoch

    """
    # config file for path mapping insde the archive
    save_config = {}
    # files to write to archive and delete afterwards
    del_files = []

    # save model to hdf5
    if model is not None:
        # temporary filename
        _curr_file = file.replace("chain", "model")
        # serialize to temporary file
        chainer.serializers.save_hdf5(_curr_file, model)
        # add to config (without path to navigate inside archive)
        save_config["model"] = os.path.basename(_curr_file)
        # append to files to process
        del_files.append(_curr_file)

    # save all optimizers to hdf5
    if optimizers is not None:
        # dict for mapping optimizer names to files
        optim_config = {}
        for k, v in optimizers.items():
            # temporary file
            _curr_file = file.replace("chain", "optim.%s" % str(k))
            # serialize to temporary file
            chainer.serializers.save_hdf5(_curr_file, v)
            # add to optimizer config (without path to navigate inside archive)
            optim_config[k] = os.path.basename(_curr_file)
            # append to files to process
            del_files.append(_curr_file)

        # add optimizer path mapping to config
        save_config["optimizers"] = optim_config

    # add epoch to config
    if epoch is not None:
        save_config["epoch"] = epoch
    # temporary config file
    _curr_file = file.replace("chain", "config")
    # serialize config dict to temporary json config file
    with open(_curr_file, "w") as f:
        json.dump(save_config, f)
    # append to files to process
    del_files.append(_curr_file)

    # create the actual archive
    with zipfile.ZipFile(file, mode="w") as f:
        for _file in del_files:
            # write temporary file to archive and remove it afterwards
            f.write(_file, os.path.basename(_file))
            os.remove(_file)


def _deserialize_and_load(archive: zipfile.ZipFile, file: str, obj,
                          temp_dir: str):
    """
    Helper Function to temporarily extract a file from a given archive,
    deserialize the object in this file and remove the temporary file

    Parameters
    ----------
    archive : :class:`zipfile.Zipfile`
        the archive containing the file to deserialize
    file : str
        identifier specifying the file inside the archive to extract and
        deserialize
    obj : Any
        the object to load the deserialized state to. Must provide a
        `serialize` function
    temp_dir : str
        the directory the file will be temporarily extracted to

    Returns
    -------
    Any
        the object with the loaded and deserialized state

    """
    # temporary extract file
    archive.extract(file, temp_dir)
    # deserialize object
    chainer.serializers.load_hdf5(os.path.join(temp_dir, file), obj)
    # remove temporary file
    os.remove(os.path.join(temp_dir, file))
    return obj


def load_checkpoint(file, old_state: dict = None,
                    model: chainer.link.Link = None, optimizers: dict = None):
    """
    Loads a state from a given file

    Parameters
    ----------
    file : str
        string containing the path to the file containing the saved state
    old_state : dict
        dictionary containing the modules to load the states to
    model : :class:`chainer.link.Link`
        the model the state should be loaded to;
        overwrites the ``model`` key in ``old_state`` if not None
    optimizers : dict
        dictionary containing all optimizers.
        overwrites the ``optimizers`` key in ``old_state`` if not None

    Returns
    -------
    dict
        the loaded state

    """
    if old_state is None:
        old_state = {}

    if model is not None:
        old_state["model"] = model
    if optimizers is not None:
        old_state["optimizers"] = optimizers

    loaded_state = {}

    # open zip archive
    with zipfile.ZipFile(file) as f:

        # load config
        _curr_file = file.replace("chain", "config")
        # temporarily extract json file to dir
        f.extract(os.path.basename(_curr_file),
                  os.path.dirname(file))
        # load config dict
        with open(_curr_file) as _file:
            config = json.load(_file)
        # remove temporary json file
        os.remove(_curr_file)

        # load model if path is inside config
        if "model" in config:
            # open file in archive by temporary extracting it
            loaded_state["model"] = _deserialize_and_load(
                f, config["model"], old_state["model"], os.path.dirname(file))

        # load optimizers if path mapping is inside config
        if "optimizers" in config:
            loaded_state["optimizers"] = {}
            optimizer_config = config["optimizers"]

            for k, v in optimizer_config.items():
                # open file in archive by temporary extracting it
                loaded_state["optimizers"][k] = _deserialize_and_load(
                    f, v, old_state["optimizers"][k], os.path.dirname(file))

        # load epoch from config if possible
        if "epoch" in config:
            loaded_state["epoch"] = config["epoch"]

    return loaded_state
