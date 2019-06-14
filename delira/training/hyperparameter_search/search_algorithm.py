from delira.training.parameters import Parameters
from sklearn.model_selection import ParameterGrid, \
    ParameterSampler as GridSampler
from delira.utils.config import LookupConfig


class SearchAlgorithm(object):
    """
    A generic search algorithm providing general parameter search.
    The actual search behavior will be specified by the ``sampler_cls``
    argument
    """
    def __init__(self, sampler_cls, params: Parameters = None,
                 n_iter: int = None):
        """

        Parameters
        ----------
        sampler_cls :
            the actual sampler class; must be subclass of
            :class:`sklearn.model_selection.ParameterSampler`
        params : :class:`Parameters`
            the parameter object defining the fixed parameters and the search
            space
        n_iter : int
            the number of trials to sample, if None: all possible trials will
            be sampled

        """
        assert issubclass(sampler_cls, GridSampler)
        self._sampler_cls = sampler_cls
        self._sampler = None

        self._fixed_space = None
        self._evaluated_trials = []
        self._setup_done = False

        if params is not None:
            self.setup(params, n_iter)

    def setup(self, params: Parameters, n_iter: int):
        """
        Computes the ParameterGrid and instantiates the sampler

        Parameters
        ----------
        params : :class:`Parameters`
            the parameter object defining the fixed parameters and the search
            space
        n_iter : int
            the number of trials to sample, if None: all possible trials will
            be sampled

        """
        self._fixed_space = params.permute_variability_on_top().fixed
        var_space = params.permute_variability_on_top().variable

        search_space = ParameterGrid(var_space.flat(max_split_size=1))
        if n_iter is None:
            n_iter = len(search_space)

        self._sampler = self._sampler_cls(search_space, n_iter)
        self._setup_done = True

    def _rebuild_params(self, trial):
        """
        Rebuilds a :class:`Parameters` object from the sampled configuration
        and the fixed parameters

        Parameters
        ----------
        trial : dict
            the sampled parameters for the current trial

        Returns
        -------
        :class:`Parameters`
            the re-constructed parameter object. Contains the sampled and the
            fixed parameters

        """
        trial_config = LookupConfig()
        trial_config.update(trial)
        return Parameters(self._fixed_space, trial_config)

    def __iter__(self):
        for trial in self._sampler:
            yield self._rebuild_params(trial)


class GridSearchAlgorithm(SearchAlgorithm):
    """
    A GridSearch Implementation
    """
    def __init__(self, params: Parameters = None, n_iter: int = None):
        """

        Parameters
        ----------
        params : :class:`Parameters`
            the parameter object defining the fixed parameters and the search
            space
        n_iter : int
            the number of trials to sample, if None: all possible trials will
            be sampled

        """
        super().__init__(GridSampler, params, n_iter)
