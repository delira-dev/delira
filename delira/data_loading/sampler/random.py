from delira.data_loading.sampler.abstract import AbstractSampler
import numpy as np


class RandomSampler(AbstractSampler):
    """
    A Generic Random Sampler
    """

    def __init__(self, indices, replacement=False, num_samples=None):
        """

        Parameters
        ----------
        indices : list
            the indices containing the classes to sample from
        replacement : bool
            whether to sample with or without replacement
        num_samples : int
            the number of samples to provide. Must only be specified
            if :param:`replacement` is True; If not specified, it defaults to
            the number of samples present in :param:`indices`
        """
        super().__init__(indices)

        if replacement and num_samples is None:
            num_samples = len(self._indices)

        self._replacement = replacement
        self._num_samples = num_samples

    def __iter__(self):
        """
        Returns an iterator returning random samples

        Returns
        -------
        Iterator
            an iterator returning random samples

        """
        n = len(self._indices)

        if self._replacement:
            return iter(np.random.randint(n, size=self._num_samples).tolist())

        possible_samples = np.arange(n)
        np.random.shuffle(possible_samples)

        return iter(possible_samples)

    def __len__(self):
        """
        Defines the length of the sampler

        Returns
        -------
        int
            the number of samples
        """
        if self._replacement:
            return self._num_samples
        else:
            return super().__len__()


class RandomSamplerNoReplacement(RandomSampler):
    """
    A Random Sampler without replacement
    """

    def __init__(self, indices):
        """

        Parameters
        ----------
        indices : list
            the indices containing the classes to sample from

        """
        super().__init__(indices, False, None)


class RandomSamplerWithReplacement(RandomSampler):
    """
    A Random Sampler With Replacement
    """

    def __init__(self, indices, num_samples=None):
        """

        Parameters
        ----------
        indices : list
            the indices containing the classes to sample from
        num_samples : int
            number of samples to provide, if not specified: defaults to the
            amount values given in :param:`indices`

        """
        super().__init__(indices, True, num_samples)
