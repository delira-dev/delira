from inspect import signature as get_signature
from sklearn.base import BaseEstimator

from delira.models.abstract_network import AbstractNetwork


class SklearnEstimator(AbstractNetwork):
    """
    Wrapper Class to wrap all ``sklearn`` estimators and provide delira
    compatibility
    """

    def __init__(self, module: BaseEstimator):
        """

        Parameters
        ----------
        module : :class:`sklearn.base.BaseEstimator`
            the module to wrap
        """

        super().__init__()

        self.module = module

        # forwards methods to self.module if necessary

        for key in ["fit", "partial_fit", "predict"]:
            if hasattr(self.module, key):
                setattr(self, key, getattr(self.module, key))

        # if estimator is build dynamically based on input, classes have to
        # be passed at least at first time (we pass it every time), because
        # not every class is present in  every batch
        # variable is initialized here, but feeded during the training
        if (self.iterative_training and "classes" in get_signature(
                self.partial_fit).parameters):
            self.classes = None

    def __call__(self, *args, **kwargs):
        """
        Calls ``self.predict`` with args and kwargs

        Parameters
        ----------
        *args :
            positional arguments of arbitrary number and type
        **kwargs :
            keyword arguments of arbitrary number and type

        Returns
        -------
        dict
            dictionary containing the predictions under key 'pred'

        """
        return {"pred": self.predict(*args, **kwargs)}

    @property
    def iterative_training(self):
        """
        Property indicating, whether a the current module can be
        trained iteratively (batchwise)

        Returns
        -------
        bool
            True: if current module can be trained iteratively
            False: else

        """
        return hasattr(self, "partial_fit")

    @staticmethod
    def prepare_batch(batch: dict, input_device, output_device):
        """
        Helper Function to prepare Network Inputs and Labels (convert them to
        correct type and shape and push them to correct devices)

        Parameters
        ----------
        batch : dict
            dictionary containing all the data
        input_device : Any
            device for module inputs (will be ignored here; just given for
            compatibility)
        output_device : Any
            device for module outputs (will be ignored here; just given for
            compatibility)

        Returns
        -------
        dict
            dictionary containing data in correct type and shape and on correct
            device

        """

        new_batch = {"X": batch["data"].reshape(batch["data"].shape[0], -1)}
        if "label" in batch:
            new_batch["y"] = batch["label"].ravel()

        return new_batch

    @staticmethod
    def closure(model, data_dict: dict, optimizers: dict, losses: dict,
                iter_num: int, fold=0, **kwargs):
        """
        default closure method to do a single training step;
        Could be overwritten for more advanced models

        Parameters
        ----------
        model : :class:`SkLearnEstimator`
            trainable model
        data_dict : dict
            dictionary containing the data
        optimizers : dict
            dictionary of optimizers to optimize model's parameters;
            ignored here, just passed for compatibility reasons
        losses : dict
            dict holding the losses to calculate errors;
            ignored here, just passed for compatibility reasons
        iter_num: int
            the number of of the current iteration in the current epoch;
            Will be restarted at zero at the beginning of every epoch
        fold : int
            Current Fold in Crossvalidation (default: 0)
        **kwargs:
            additional keyword arguments

        Returns
        -------
        dict
            Loss values (with same keys as input dict losses; will always
            be empty here)
        dict
            dictionary containing all predictions

        """

        if model.iterative_training:
            fit_fn = model.partial_fit

        else:
            fit_fn = model.fit

        if hasattr(model, "classes"):
            # classes must be specified here, because not all classes
            # must be present in each batch and some estimators are build
            # dynamically
            fit_fn(**data_dict, classes=model.classes)
        else:
            fit_fn(**data_dict)

        preds = model(data_dict["X"])

        return {}, preds
