from multiprocessing import Pool
from delira import get_current_debug_mode

# class Reporter(object):
#     __HIGHEST_KEYWORDS = ("highest", "max")
#     __LOWEST_KEYWORDS = ("lowest", "min")
#
#     def __init__(self, score_key, score_mode):
#         self._trials = []
#         self._best_trial = None
#         self._current_trial = None
#         self._score_key = score_key
#         self._score_mode = score_mode.lower()
#         self._scheduler
#
#     def start_new_trial(self, config):
#         trial = Trial(config, self._score_key)
#         self._current_trial = trial
#
#     def end_trial(self):
#         if self._best_trial is None:
#             best_trial = self._current_trial
#         elif self._current_is_best():
#             best_trial = self._current_trial
#         else:
#             best_trial = self._best_trial
#
#         self._best_trial = best_trial
#
#     def __call__(self):
#
#
#     def _current_is_best(self):
#         if self._score_mode in self.__HIGHEST_KEYWORDS:
#             return self._current_trial > self._best_trial
#         elif self._score_mode in self.__LOWEST_KEYWORDS:
#             return self._current_trial < self._best_trial
#
#         raise ValueError("Invalid Score Mode: %s" % str(self._score_mode))
#
#
# class Trial(list):
#
#     def __init__(self, config: dict, score_key: str, result=None):
#         super().__init__()
#         if result is None:
#             result = {}
#         assert config
#         self._config = config
#
#         for key, val in result.items():
#             if not isinstance(val, list):
#                 result[key] = [val]
#
#         self._result = result
#
#         assert score_key
#         assert score_key in self._result
#
#         self._score_key = score_key
#
#     def __getitem__(self, index, name=None):
#         if name is None:
#             name = self._score_key
#         return self._result[name][index]
#
#     def __setitem__(self, index, value, name=None):
#         if name is None:
#             name = self._score_key
#
#         self._result[name][index] = value
#
#     def append(self, other: dict) -> None:
#         for k, v in other.items():
#             if isinstance(self._result[k], list):
#                 self._result[k].append(v)
#             else:
#                 self._result[k] = [v]
#
#     @property
#     def score(self):
#         return self._result[self._score_key]
#
#     @staticmethod
#     def __extract_score_from_other(other):
#         if not isinstance(other, (int, float)):
#             other = other.score
#         return other
#
#     def __gt__(self, other):
#         return self.score > self.__extract_score_from_other(other)
#
#     def __ge__(self, other):
#         return self.score >= self.__extract_score_from_other(other)
#
#     def __lt__(self, other):
#         return self.score < self.__extract_score_from_other(other)
#
#     def __le__(self, other):
#         return self.score <= self.__extract_score_from_other(other)
#
#     def __eq__(self, other):
#         return self.score == self.__extract_score_from_other(other)
#


class _DebugPool:
    """
    A Utility class to use when multiprocessing should be disabled.
    Instead of parallel mapping (as :class:`multiprocessing.Pool` does), this
    class does sequential mapping
    """
    def map(self, function, argument_list):
        """
        Sequentially maps a given function to a list of arguments

        Parameters
        ----------
        function : function
            the function to map to all arguments
        argument_list : list or tuple
            an iterable containing all arguments, this function should be
            applied to

        Returns
        -------
        tuple
            the result of all function calls

        """
        return map(function, argument_list)

    def map_async(self, function, argument_list):
        """
        Sequentially maps a given function to a list of arguments

        Parameters
        ----------
        function : function
            the function to map to all arguments
        argument_list : list or tuple
            an iterable containing all arguments, this function should be
            applied to

        Returns
        -------
        tuple
            the result of all function calls

        """
        return self.map(function, argument_list)

    def __enter__(self):
        pass

    def __exit__(self, *args, **kwargs):
        pass


class DebugSafePool:
    """
    Utility Class to dispatch to sequential mapping if debug mode is activated
    """
    def __init__(self, n_processes=None, *args, **kwargs):
        """

        Parameters
        ----------
        n_processes : int
            the number of processes to use for parallel mapping
        *args :
            additional positional arguments
        **kwargs :
            additional keyword arguments
        """
        if not get_current_debug_mode():
            self._pool = Pool(n_processes, *args, **kwargs)
        else:
            self._pool = _DebugPool()

    def map(self, function, arg_list, *args):
        """
        Dispatches the mapping to either sequential or parallel mapping

        Parameters
        ----------
        function : function
            the function to apply
        arg_list : tuple or list
            the arguments, the function should be applied to
        *args
            additional argument sets

        Returns
        -------
        tuple
            the results of function mapping

        """
        return self._pool.map(function, type(arg_list)((*arg_list, *args)))

    def map_async(self, function, arg_list, *args):
        """
        Dispatches the async mapping to either sequential sync mapping
        (no sequential asynch mapping possible) or parallel async mapping

        Parameters
        ----------
        function : function
            the function to apply
        arg_list : tuple or list
            the arguments, the function should be applied to
        *args
            additional argument sets

        Returns
        -------
        tuple
            the results of function mapping

        """
        return self._pool.map_async(function, type(arg_list)((*arg_list,
                                                              *args)))

    def __enter__(self):
        return self._pool.__enter__()

    def __exit__(self, *args, **kwargs):
        return self._pool.__exit__(*args, **kwargs)
