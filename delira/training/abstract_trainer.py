from abc import abstractmethod
import logging
import pickle

from batchgenerators.dataloading import MultiThreadedAugmenter

from .callbacks import AbstractCallback


logger = logging.getLogger(__name__)

KEYS_TO_GUARD = ["use_gpu",
                 "input_device",
                 "output_device",
                 "_callbacks"]


class AbstractNetworkTrainer(object):
    """
    Defines an abstract API for Network Trainers

    See Also
    --------
    :class:`PyTorchNetworkTrainer`

    """

    def __init__(self, fold=0, callbacks=[]):
        """

        Parameters
        ----------
        fold : int
            current fold in (default: 0)
        callbacks : list
            list of callbacks to register

        """
        self._callbacks = []
        self._fold = fold

        for cbck in callbacks:
            self.register_callback(cbck)

    @abstractmethod
    def _setup(self, *args, **kwargs):
        """
        Defines the actual Trainer Setup

        Parameters
        ----------
        *args :
            positional arguments
        **kwargs :
            keyword arguments

        Raises
        ------
        NotImplementedError
            If not overwritten by subclass

        """
        raise NotImplementedError()

    @abstractmethod
    def _at_training_begin(self, *args, **kwargs):
        """
        Defines the behaviour at beginnig of the training

        Parameters
        ----------
        *args :
            positional arguments
        **kwargs :
            keyword arguments

        Raises
        ------
        NotImplementedError
            If not overwritten by subclass

        """
        raise NotImplementedError()

    @abstractmethod
    def _at_training_end(self, *args, **kwargs):
        """
        Defines the behaviour at the end of the training

        Parameters
        ----------
        *args :
            positional arguments
        **kwargs :
            keyword arguments

        Raises
        ------
        NotImplementedError
            If not overwritten by subclass

        """
        raise NotImplementedError()

    @abstractmethod
    def _at_epoch_begin(self, *args, **kwargs):
        """
        Defines the behaviour at beginnig of each epoch

        Parameters
        ----------
        *args :
            positional arguments
        **kwargs :
            keyword arguments

        Raises
        ------
        NotImplementedError
            If not overwritten by subclass

        """
        raise NotImplementedError()

    @abstractmethod
    def _at_epoch_end(self, *args, **kwargs):
        """
        Defines the behaviour at the end of each epoch

        Parameters
        ----------
        *args :
            positional arguments
        **kwargs :
            keyword arguments

        Raises
        ------
        NotImplementedError
            If not overwritten by subclass

        """
        raise NotImplementedError()

    @abstractmethod
    def _train_single_epoch(self, batchgen: MultiThreadedAugmenter, epoch):
        """
        Defines a routine to train a single epoch

        Parameters
        ----------
        batchgen : MultiThreadedAugmenter
            generator holding the batches
        epoch : int
            current epoch

        Raises
        ------
        NotImplementedError
            If not overwritten by subclass

        """
        raise NotImplementedError()

    @abstractmethod
    def train(self, num_epochs, datamgr_train, datamgr_valid=None,
              val_score_key=None, val_score_mode='highest'):
        """
        Defines a routine to train a specified number of epochs

        Parameters
        ----------
        num_epochs : int
            number of epochs to train
        datamgr_train : DataManager
            the datamanager holding the train data
        datamgr_valid : DataManager
            the datamanager holding the validation data (default: None)
        val_score_key : str
            the key specifying which metric to use for validation
            (default: None)
        val_score_mode : str
            key specifying what kind of validation score is best

        Raises
        ------
        NotImplementedError
            If not overwritten by subclass

        """
        raise NotImplementedError()

    @abstractmethod
    def predict(self, batchgen, batchsize=None):
        """
        Defines a rotine to predict data obtained from a batchgenerator

        Parameters
        ----------
        batchgen : MultiThreadedAugmenter
            Generator Holding the Batches
        batchsize : Artificial batchsize (sampling will be done with batchsize
                    1 and sampled data will be stacked to match the artificial
                    batchsize)(default: None)

        Raises
        ------
        NotImplementedError
            If not overwritten by subclass

        """
        raise NotImplementedError()

    @property
    def fold(self):
        """
        Get current fold

        Returns
        -------
        int
            current fold

        """
        return self._fold

    @fold.setter
    def fold(self, fold):
        """
        Set the current fold

        Parameters
        ----------
        fold : int
            new fold

        Raises
        ------
        ValueError
            if `fold` is not covertable to :obj:`int`

        """
        try:
            self._fold = int(fold)

        except ValueError as e:
            logger.error(e)
            raise e

    def __setattr__(self, key, value):
        """
        Set attributes and guard specific attributes after they have been set
        once

        Parameters
        ----------
        key : str
            the attributes name
        value : Any
            the value to set

        Raises
        ------
        PermissionError
            If attribute which should be set is guarded

        """

        # check if key has been set once
        if key in KEYS_TO_GUARD and hasattr(self, key):
            raise PermissionError("%s should not be overwritten after "
                                  "it has been set once" % key)
        else:
            super().__setattr__(key, value)

    def register_callback(self, callback: AbstractCallback):
        """
        Register Callback to Trainer

        Parameters
        ----------
        callback : :class:`AbstractCallback`
            the callback to register

        Raises
        ------
        AssertionError
            `callback` is not an instance of :class:`AbstractCallback` and has
            not both methods ['at_epoch_begin', 'at_epoch_end']

        """
        assertion_str = "Given callback is not valid; Must be instance of " \
                        "AbstractCallback or provide functions " \
                        "'at_epoch_begin' and 'at_epoch_end'"

        assert isinstance(callback, AbstractCallback) or \
            (hasattr(callback, "at_epoch_begin")
             and hasattr(callback, "at_epoch_end")), assertion_str

        self._callbacks.append(callback)

    def save_state(self, file_name, *args, **kwargs):
        """
        saves the current state

        Parameters
        ----------
        file_name : str
            filename to save the state to
        *args :
            positional arguments
        **kwargs :
            keyword arguments

        """
        with open(file_name, "wb") as f:
            pickle.dump(vars(self), f, *args, **kwargs)

    @staticmethod
    def load_state(file_name, *args, **kwargs):
        """
        Loads the new state from file

        Parameters
        ----------
        file_name : str
            the file to load the state from
        *args :
            positional arguments
        **kwargs : keyword arguments

        Returns
        -------
        dict
            new state

        """
        with open(file_name, "rb") as f:
            new_state = pickle.load(f, *args, **kwargs)

        return new_state

    def _update_state(self, new_state):
        """
        Update the state from a given new state

        Parameters
        ----------
        new_state : dict
            new state to update internal state from

        Returns
        -------
        :class:`AbstractNetworkTrainer`
            the trainer with a modified state

        """
        for key, val in new_state.items():
            if (key.startswith("__") and key.endswith("__")):
                continue

            try:
                setattr(self, key, val)

            except PermissionError:
                logger.error("Trying to overwrite attribute %s of "
                             "NetworkTrainer, which is not allowed!" % key)

        return self

    def update_state(self, file_name, *args, **kwargs):
        """
        Update internal state from a loaded state

        Parameters
        ----------
        file_name : str
            file containing the new state to load
        *args :
            positional arguments
        **kwargs :
            keyword arguments

        Returns
        -------
        :class:`AbstractNetworkTrainer`
            the trainer with a modified state

        """
        self._update_state(self.load_state(file_name, *args, **kwargs))

    @staticmethod
    def _is_better_val_scores(old_val_score, new_val_score,
                              mode='highest'):
        """
        Check whether the new val score is better than the old one
        with respect to the optimization goal

        Parameters
        ----------
        old_val_score :
            old validation score
        new_val_score :
            new validation score
        mode: str
            String to specify whether a higher or lower validation score is
            optimal; must be in ['highest', 'lowest']

        Returns
        -------
        bool
            True if new score is better, False otherwise
        """

        assert mode in ['highest', 'lowest'], "Invalid Comparison Mode"

        if mode == 'highest':
            return new_val_score > old_val_score
        elif mode == 'lowest':
            return new_val_score < old_val_score
