import typing
import torch
from collections import OrderedDict
from trixi.util import Config
import pickle
import yaml
import os
import shutil
import zipfile

IGNORE_EXPORT_KEYS = ["_criterions", "criterions", "_metrics", "metrics",
                      "optimizer_cls", "lr_sched_cls"]


class Hyperparameters(Config):
    """
    Class containing all HyperParameters
    """

    def __init__(self, batch_size, num_epochs, optimizer_cls, optimizer_params,
                 criterions:typing.Union[typing.List, typing.Dict],
                 lr_sched_cls, lr_sched_params,
                 metrics: typing.Union[typing.List, typing.Dict], **kwargs):
        """

        Parameters
        ----------
        batch_size : int
            Number of Samples per Batch
        num_epochs : int
            Number of Epochs to train
        optimizer_cls :
            Optimizer Class (must be subclass of torch.optim.Optimizer)
        optimizer_params :
            Parameters for optimizer construcion
        criterions : list or dict
            Training criterions
            If List is given: dict will be created with classnames as keys
        lr_sched_cls :
            Learning rate scheduling class (must be subclass of
            `AbstractCallback`)
        lr_sched_params :
            Parameters for lr scheduler construction
        metrics : list or dict
            Metrics for network performance monitoring
            If List is given: dict will be created with classnames as keys
        **kwargs :
            additional keyword arguments

        """
        self._criterions = None
        self._metrics = None

        super().__init__(batch_size=batch_size,
                         num_epochs=num_epochs,
                         optimizer_cls=optimizer_cls,
                         optimizer_params=optimizer_params,
                         _criterions=criterions,
                         lr_sched_cls=lr_sched_cls,
                         lr_sched_params=lr_sched_params,
                         _metrics=metrics,
                         **kwargs)

    @classmethod
    def from_file(cls, file_name):
        """
        Classmethod to create Hyper Parameters from file

        Parameters
        ----------
        file_name : str
            file to create hyperparameters from

        Returns
        -------
        :class:`Hyperparameters`
            hyperparameters with arguments loaded from file

        """
        return cls(**super(cls).__init__(file_name))

    @property
    def criterions(self):
        """
        Returns criterions as dict

        Returns
        -------
        dict
            criterions

        """

        if isinstance(self._criterions, dict) or isinstance(self._criterions,
                                                            OrderedDict):
            crits = self._criterions
        else:
            crits = {}
            for crit in self._criterions:
                crits[crit.__class__.__name__] = crit

        return crits

    @property
    def metrics(self):
        """
        Returns metrics as dict

        Returns
        -------
        dict
            metrics

        """
        if isinstance(self._metrics, dict) or isinstance(self._metrics,
                                                         OrderedDict):
            metrics = self._metrics
        else:
            metrics = {}
            for metr in self._metrics:
                metrics[metr.__class__.__name__] = metr

        return metrics

    @criterions.setter
    def criterions(self, criterions):
        """
        Updates Criterions

        Parameters
        ----------
        criterions : list
            the new criterions

        """
        self._criterions = criterions

    @metrics.setter
    def metrics(self, metrics):
        """
        Updates Metrics

        Parameters
        ----------
        metrics :
            the new metrics

        """
        self._metrics = metrics

    def _to_yaml(self, yaml_file_path):
        """
        Helper Function to export non-binary parts to YAML file

        Parameters
        ----------
        yaml_file_path : str
            path to yaml file

        """
        yaml_dict = {}
        for k, v in vars(self).items():
            if not (k.startswith("_") or k == "metrics" or k == "criterions"
                    or k == "optimizer_cls" or k == "lr_sched_cls"):
                yaml_dict[k] = v
            if k.endswith("_params"):
                yaml_dict[k] = dict(v)

            else:
                yaml_dict[k] = v.__class__.__name__

        with open(yaml_file_path, "w") as f:
            yaml.dump({"HyperParameters": yaml_dict}, f,
                      default_flow_style=False)

    def _pickle_class_args(self, pickle_file_path):
        """
        Helper Function to export binary parts to pickle file

        Parameters
        ----------
        pickle_file_path : str
            Path to Pickle File

        """
        pickle_dict = {"criterions": self.criterions,
                       "metrics": self.metrics,
                       "optimizer_cls": self.optimizer_cls,
                       "lr_sched_cls": self.lr_sched_cls}

        with open(pickle_file_path, "wb") as f:
            pickle.dump(pickle_dict, f)

    @classmethod
    def from_files(cls, export_root_path, zipped=True):
        """
        Create an instance of `Hyperparameters` class from files

        Parameters
        ----------
        export_root_path : str
            path containing the hyperparameter files or zip
        zipped : bool
            whether the hyperparameter files are zipped (default: True)

        Returns
        -------
        :class:`Hyperparameters`
            Instance with attributes loaded from files

        """

        if zipped:
            zip_path = os.path.join(export_root_path, "hyper_params.zip")

            with zipfile.ZipFile(zip_path) as zfile:
                with zfile.open("binary_components.pkl", "rb") as f:
                    pickle_dict = pickle.load(f)

                with zfile.open("hyper_params.yml") as f:
                    yaml_dict = yaml.load(f)["HyperParameters"]

        else:
            with open(os.path.join(export_root_path, "HyperParams",
                                   "binary_components.pkl"), "rb") as f:
                pickle_dict = pickle.load(f)

            with open(os.path.join(export_root_path, "HyperParams",
                                   "hyper_params.yml")) as f:
                yaml_dict = yaml.load(f)["HyperParameters"]

        return cls(**pickle_dict, **yaml_dict)

    def export_to_files(self, export_root_path, zip=True):
        """
        Export class to YAML and pickle files

        Parameters
        ----------
        export_root_path : str
            root path to export the class to
        zip : bool
            whether to zip the files (default: True)

        """
        pickle_file = "binary_components.pkl"
        yaml_file = "hyper_params.yml"

        self._to_yaml(yaml_file)
        self._pickle_class_args(pickle_file)

        if zip:
            zip_file = os.path.join(export_root_path, "hyper_params.zip")
            with zipfile.ZipFile(zip_file, mode="w") as f:
                f.write(yaml_file)
                f.write(pickle_file)

                os.remove(yaml_file)
                os.remove(pickle_file)

        else:

            pickle_file = os.path.join(export_root_path, pickle_file)
            yaml_file = os.path.join(export_root_path, yaml_file)

            os.makedirs(os.path.join(export_root_path, "HyperParams"),
                        exist_ok=True)
            shutil.move(yaml_file, os.path.join(export_root_path,
                                                "HyperParams"))
            shutil.move(pickle_file, os.path.join(export_root_path,
                                                  "HyperParams"))

    def __str__(self):
        s = "Hyperparameters:\n"
        for k, v in vars(self).items():
            s += "\t{} = {}\n".format(k, v)
        return s
