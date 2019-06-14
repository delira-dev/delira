import pickle
from copy import deepcopy, copy

import yaml

from ..utils import LookupConfig


class Parameters(LookupConfig):
    """
    Class Containing all variable and fixed parameters for training and model
    instantiation

    See Also
    --------
    ``trixi.util.Config``

    """

    def __init__(self, fixed_params=None,
                 variable_params=None):
        """

        Parameters
        ----------
        fixed_params : dict
            fixed parameters (are not variated using hyperparameter search)
        variable_params: dict
            variable parameters (can be variated by a hyperparameter search)
        """

        if variable_params is None:
            variable_params = {"model": {},
                               "training": {}}
        if fixed_params is None:
            fixed_params = {"model": {},
                            "training": {}}

        super().__init__(fixed=fixed_params,
                         variable=variable_params)

    def permute_hierarchy(self):
        """
        switches hierarchy

        Returns
        -------
        Parameters
            the class with a permuted hierarchy

        Raises
        ------
        AttributeError
            if no valid hierarchy is found

        """

        if self.variability_on_top:
            fixed = self.pop("fixed")
            variable = self.pop("variable")

            model = {
                "fixed": fixed.pop("model"),
                "variable": variable.pop("model")}
            training = {"fixed": fixed.pop("training"),
                        "variable": variable.pop("training")}

            self.model = model
            self.training = training

        elif self.training_on_top:
            model = self.pop("model")
            training = self.pop("training")

            fixed = {
                "model": model.pop("fixed"),
                "training": training.pop("fixed")}
            variable = {
                "model": model.pop("variable"),
                "training": training.pop("variable")}

            self.fixed = fixed
            self.variable = variable

        else:
            return AttributeError(
                "%s must either have keys ('model', 'training') or "
                "('fixed', 'variable')" % self.__class__.__name__)

        return self

    def permute_training_on_top(self):
        """
        permutes hierarchy in a way that the training-model hierarchy is on top

        Returns
        -------
        Parameters
            Parameters with permuted hierarchy

        """

        if self.training_on_top:
            return self
        else:
            return self.permute_hierarchy()

    def permute_variability_on_top(self):
        """
        permutes hierarchy in a way that the training-model hierarchy is on top

        Returns
        -------
        Parameters
            Parameters with permuted hierarchy

        """
        if self.variability_on_top:
            return self
        else:
            return self.permute_hierarchy()

    @property
    def hierarchy(self):
        """
        Returns the current hierarchy

        Returns
        -------
        str
            current hierarchy

        """

        if self.variability_on_top:
            hierarchy = "variability\n|\n->\ttraining"
        elif self.training_on_top:
            hierarchy = "training\n|\n->\tvariability"

        else:
            hierarchy = "no valid hierarchy"

        return hierarchy

    def permute_to_hierarchy(self, hierarchy: str):
        """
        Permute hierarchy to match the specified hierarchy

        Parameters
        ----------
        hierarchy : str
            target hierarchy

        Raises
        ------
        ValueError
            Specified hierarchy is invalid

        Returns
        -------
        Parameters
            parameters with proper hierarchy

        """

        if hierarchy == "variability\n|\n->\ttraining":
            return self.permute_training_on_top()
        elif hierarchy == "training\n|\n->\tvariability":
            return self.permute_variability_on_top()

        else:
            raise ValueError("Invalid Hierarchy: %s" % hierarchy)

    @property
    def variability_on_top(self):
        """
        Return whether the variability is on top

        Returns
        -------
        bool
            whether variability is on top

        """

        return hasattr(self, "fixed") and hasattr(self, "variable")

    @property
    def training_on_top(self):
        """
        Return whether the training hierarchy is on top

        Returns
        -------
        bool
            whether training is on top

        """
        return hasattr(self, "model") and hasattr(self, "training")

    def save(self, filepath: str):
        """
        Saves class to given filepath (YAML + Pickle)

        Parameters
        ----------
        filepath : str
            file to save data to

        """

        if not (filepath.endswith(".yaml") or filepath.endswith(".yml")):
            filepath = filepath + ".yml"

        try:
            with open(filepath, "w") as f:
                yaml.dump(self.permute_variability_on_top(), f)

        except TypeError:
            pass

        finally:
            with open(filepath.replace(".yaml", "").replace(".yml", ""),
                      "wb") as f:
                pickle.dump(self, f)

    def update(self, dict_like, deep=False, ignore=None,
               allow_dict_overwrite=True):
        """Update entries in the Parameters

        Parameters
        ----------
        dict_like : dict
            Update source
        deep : bool
            Make deep copies of all references in the source.
        ignore : Iterable
            Iterable of keys to ignore in update
        allow_dict_overwrite : bool
            Allow overwriting with dict.
            Regular dicts only update on the highest level while we recurse
            and merge Configs. This flag decides whether it is possible to
            overwrite a 'regular' value with a dict/Config at lower levels.
            See examples for an illustration of the difference

        Examples
        --------
        The following illustrates the update behaviour if
        :obj:allow_dict_overwrite is active. If it isn't, an AttributeError
        would be raised, originating from trying to update "string"::

            config1 = Config(config={
                "lvl0": {
                    "lvl1": "string",
                    "something": "else"
                }
            })

            config2 = Config(config={
                "lvl0": {
                    "lvl1": {
                        "lvl2": "string"
                    }
                }
            })

            config1.update(config2, allow_dict_overwrite=True)

            >>>config1
            {
                "lvl0": {
                    "lvl1": {
                        "lvl2": "string"
                    },
                    "something": "else"
                }
            }

        """
        empty = self.variability_on_top == self.training_on_top
        if not empty:
            variability_on_top = self.variability_on_top

            if variability_on_top:
                if isinstance(dict_like, Parameters):
                    dict_like_variability_on_top = dict_like.variability_on_top
                    dict_like = dict_like.permute_variability_on_top()
                else:
                    if ("fixed" not in dict_like.keys() and "variable" not in
                            dict_like.keys()):
                        raise RuntimeError("Unsafe to Update from dict with "
                                           "another structre as current "
                                           "parameters")

            else:
                if isinstance(dict_like, Parameters):
                    dict_like_variability_on_top = dict_like.variability_on_top
                    dict_like = dict_like.permute_training_on_top()
                else:
                    if ("model" not in dict_like.keys() and "training" not in
                            dict_like.keys()):
                        raise RuntimeError("Unsafe to Update from dict with "
                                           "another structre as current "
                                           "parameters")

        super().update(dict_like=dict_like, deep=deep,
                       ignore=ignore,
                       allow_dict_overwrite=allow_dict_overwrite)

        if not empty and isinstance(dict_like, Parameters):
            # restore original permutation of dict_like
            if variability_on_top and not dict_like_variability_on_top:
                # dict_like changed to variability_on_top
                dict_like.permute_training_on_top()
            elif not variability_on_top and dict_like_variability_on_top:
                # dict_like changed to training_on_top
                dict_like.permute_variability_on_top()

    def __str__(self):
        """
        String Representation of class

        Returns
        -------
        str
            string representation
        """

        s = "Parameters:\n"
        for k, v in vars(self).items():
            try:
                s += "\t{} = {}\n".format(k, v)
            except TypeError:
                s += "\t{} = {}\n".format(k, v.__class__.__name__)
        return s

    def __copy__(self):
        """
        Enables shallow copy

        Returns
        -------
        :class:`Parameters`
            copied parameters

        """

        var_top = self.variability_on_top

        _params = Parameters()
        _params.update(copy(dict(self.permute_variability_on_top())))

        if var_top:
            return _params.permute_variability_on_top()
        else:
            # restore original perumation
            self.permute_training_on_top()
            return _params.permute_training_on_top()

    def __deepcopy__(self, memo):
        """
        Enables deepcopy

        Returns
        -------
        :class:`Parameters`
            deepcopied parameters

        """

        var_top = self.variability_on_top

        _params = Parameters()
        _params.update(deepcopy(dict(self.permute_variability_on_top()),
                                memo=memo))

        if var_top:
            return _params.permute_variability_on_top()
        else:
            # restore original perumation
            self.permute_training_on_top()
            return _params.permute_training_on_top()
