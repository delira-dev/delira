from ..utils import LookupConfig
import pickle
import yaml

class Parameters(LookupConfig):
    """
    Class Containing all variable and fixed parameters for training and model 
    instantiation

    See Also
    --------
    ``trixi.util.Config``

    """

    def __init__(self, fixed_params={"model": {},
                                     "training": {}},
                 variable_params={"model": {},
                                  "training": {}}):
        """

        Parameters
        ----------
        fixed_params : dict
            fixed parameters (are not variated using hyperparameter search)
        variable_params: dict
            variable parameters (can be variated by a hyperparameter search)
        """

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

    
