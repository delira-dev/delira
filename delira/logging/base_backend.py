
from queue import Empty
from abc import abstractmethod, ABCMeta
from threading import Event
from queue import Queue
import warnings

_FUNCTIONS_WITHOUT_STEP = ("graph_pytorch", "graph_tf", "graph_onnx",
                           "embedding")

# Deprecated Keys with their future alternative
_DEPRECATED_KEYS = {"img": "image", "picture": "image", "imgs": "images",
                    "pictures": "images", "bounding_boxes": "image_with_boxes",
                    "bboxes": "image_with_boxes", "value": "scalar",
                    "values": "scalar", "hist": "histogram", "fig": "figure",
                    "sound": "audio", "pr": "pr_curve", "curve": "line",
                    "hm": "heatmap"}


class BaseBackend(object, metaclass=ABCMeta):
    """
    The basic Logging Backend, Provides an abstract interface to log
    different value types and some keyword mappings
    """

    class FigureManager:
        """
        A Figure Manager, which creates a figure during entrance and pushes
        the figure to logging writer during exit
        """

        def __init__(self, push_fn, figure_kwargs: dict, push_kwargs: dict):
            """

            Parameters
            ----------
            push_fn : function
                A function accepting a figure and some keyword arguments
                to push it to the logging writer
            figure_kwargs : dict
                dictionary containing all keyword arguments to create the
                figure
            push_kwargs : dict
                dictionary containing all keyword arguments to push the figure
                to the loggging writer
            """
            self._push_fn = push_fn
            self._figure_kwargs = figure_kwargs
            self._push_kwargs = push_kwargs
            self._fig = None

        def __enter__(self):
            """
            Function to be executed during context-manager entrance;
            Will create a figure with the figure kwargs

            """
            from matplotlib.pyplot import figure
            self._fig = figure(**self._figure_kwargs)

        def __exit__(self, *args):
            """
            Function to be executed during context-manager exit;
            Will push the figure to the logging writer and destroy it
            afterwards

            Parameters
            ----------
            *args :
                arbitrary positional arguments; Necessary to be compatible
                with other context managers, but not used in this one

            """
            from matplotlib.pyplot import close
            self._push_fn(figure=self._fig, **self._push_kwargs)

            close(self._fig)
            self._fig = None

    def __init__(self, abort_event: Event = None, queue: Queue = None):
        """

        Parameters
        ----------
        abort_event : :class:`threading.Event`
            the event to signalize, when the logger must be destroyed
        queue : :class:`queue.Queue`
            the queue to enqueue all tuples of mapped functions and the
            corresponding arguments before their execution

        """
        super().__init__()
        self.KEYWORD_FN_MAPPING = {}

        self.daemon = True

        self._queue = queue
        self._abort_event = abort_event
        self._global_steps = {}
        # create Keyword mapping
        self.KEYWORD_FN_MAPPING.update(**{
            "image": self._image,
            "img": self._image,
            "picture": self._image,
            "images": self._images,
            "imgs": self._images,
            "pictures": self._images,
            "image_with_boxes": self._image_with_boxes,
            "bounding_boxes": self._image_with_boxes,
            "bboxes": self._image_with_boxes,
            "scalar": self._scalar,
            "value": self._scalar,
            "scalars": self._scalars,
            "values": self._scalars,
            "histogram": self._histogram,
            "hist": self._histogram,
            "figure": self._figure,
            "fig": self._figure,
            "audio": self._audio,
            "sound": self._audio,
            "video": self._video,
            "text": self._text,
            "graph_pytorch": self._graph_pytorch,
            "graph_tf": self._graph_tf,
            "graph_onnx": self._graph_onnx,
            "embedding": self._embedding,
            "pr_curve": self._pr_curve,
            "pr": self._pr_curve,
            "scatter": self._scatter,
            "line": self._line,
            "curve": self._line,
            "stem": self._stem,
            "heatmap": self._heatmap,
            "hm": self._heatmap,
            "bar": self._bar,
            "boxplot": self._boxplot,
            "surface": self._surface,
            "contour": self._contour,
            "quiver": self._quiver,
            # "mesh": self._mesh
        })

    def _log_item(self):
        """
        Internal helper function to log an item of the queue

        Raises
        ------
        ValueError
            if the item to log is not a dict

        """
        # get item from dict
        process_item = self._queue.get(timeout=0.001)
        # log item if item is dict
        if isinstance(process_item, dict):

            for key, val in process_item.items():
                # raise DeprecationWarning for deprecated keys
                if key in _DEPRECATED_KEYS:
                    warnings.warn("The Key %s is deprecated and will"
                                  " be removed in the next release. "
                                  "Please use %s instead!"
                                  % (key, _DEPRECATED_KEYS[key]),
                                  DeprecationWarning)

                # performs the actual mapping
                execute_fn = self.KEYWORD_FN_MAPPING[str(key).lower()]

                # resolve the global step
                val = self._resolve_global_step(str(key).lower(), **val)

                # execute the logging function
                self._call_exec_fn(execute_fn, val)

        # item is no dict -> raise Error
        else:
            raise ValueError("Invalid Value passed for logging: %s"
                             % str(process_item))

    def _resolve_global_step(self, key, **val):
        """
        Helper function to resolve the global step from given Arguments

        Parameters
        ----------
        key : str
            the function key to resolve the step for
        **val :
            kwargs which may contain the step information

        Returns
        -------
        int
            the global step

        Raises
        ------
        ValueError
            If no valid tag was found although a tag should exist

        """
        # check if function should be processed statically
        # (no time update possible)
        if str(key).lower() not in _FUNCTIONS_WITHOUT_STEP:

            # check for different step names
            if "tag" in val:
                tag = "tag"
            elif "main_tag" in val:
                tag = "main_tag"
            else:
                raise ValueError("No valid tag found to extract global step")

            # check if global step is given
            if "global_step" not in val or val["global_step"] is None:

                # check if tag is already part of internal global steps
                if val[tag] in self._global_steps:
                    # if already existent: increment step for given tag
                    self._global_steps[val[tag]] += 1
                    step = self._global_steps[val[tag]]

                else:
                    # if not existent_ set step for given tag to zero
                    step = 0
                    self._global_steps[val[tag]] = step

                val.update({"global_step": step})

            elif "global_step" in val:
                self._global_steps[tag] = val["global_step"]

        return val

    def run(self):
        """
        Main function which executes the logging, catches exceptions and sets
        the abortion event if necessary

        """
        try:
            self._log_item()

        except Empty:
            pass

        except Exception as e:
            self._abort_event.set()
            raise e

    def set_queue(self, queue: Queue):
        """
        Setter Function for the Queue

        Parameters
        ----------
        queue : :class:`queue.Queue`
            the new queue

        """
        self._queue = queue

    def set_event(self, event: Event):
        """
        Setter Function for the abortion event

        Parameters
        ----------
        event : :class:`threading.Event`
            the new abortion event

        """
        self._abort_event = event

    def _call_exec_fn(self, exec_fn, args):
        """
        Helper Function calling the actual  mapped function

        Parameters
        ----------
        exec_fn : function
            the function which will execute the actual logging
        args : iterable (listlike) or mapping (dictlike)
            the arguments passed to the ``exec_fn``

        Returns
        -------
        Any
            the return value obtained by the ``exec_fn``

        Raises
        ------
        TypeError
            if the given ``args`` are neither of type dict or tuple/list

        """

        if isinstance(args, dict):
            ret_val = exec_fn(**args)
        elif isinstance(args, (tuple, list)):
            ret_val = exec_fn(*args)

        else:
            raise TypeError("Invalid type for args. Must be either dict, "
                            "tuple or list, but got %s."
                            % args.__class__.__name__)

        return ret_val

    @abstractmethod
    def _image(self, *args, **kwargs):
        """
        Abstract Interface Function to log a single image

        Parameters
        ----------
        *args
            arbitrary positional arguments
        **kwargs
            arbitrary keyword arguments

        Raises
        ------
        NotImplementedError
            If not overwritten in subclass

        """
        raise NotImplementedError

    @abstractmethod
    def _images(self, *args, **kwargs):
        """
        Abstract Interface Function to log multiple images

        Parameters
        ----------
        *args
        **kwargs

        Raises
        ------
        NotImplementedError
            If not overwritten in subclass

        """
        raise NotImplementedError

    @abstractmethod
    def _image_with_boxes(self, *args, **kwargs):
        """
        Abstract Interface Function to log a single image with bounding boxes

        Parameters
        ----------
        *args
            arbitrary positional arguments
        **kwargs
            arbitrary keyword arguments

        Raises
        ------
        NotImplementedError
            If not overwritten in subclass

        """
        raise NotImplementedError

    @abstractmethod
    def _scalar(self, *args, **kwargs):
        """
        Abstract Interface Function to log a single scalar value

        Parameters
        ----------
        *args
            arbitrary positional arguments
        **kwargs
            arbitrary keyword arguments

        Raises
        ------
        NotImplementedError
            If not overwritten in subclass

        """
        raise NotImplementedError

    @abstractmethod
    def _scalars(self, *args, **kwargs):
        """
        Abstract Interface Function to log multiple scalar values

        Parameters
        ----------
        *args
            arbitrary positional arguments
        **kwargs
            arbitrary keyword arguments

        Raises
        ------
        NotImplementedError
            If not overwritten in subclass

        """
        raise NotImplementedError

    @abstractmethod
    def _histogram(self, *args, **kwargs):
        """
        Abstract Interface Function to create and log a histogram out of given
        values

        Parameters
        ----------
        *args
            arbitrary positional arguments
        **kwargs
            arbitrary keyword arguments

        Raises
        ------
        NotImplementedError
            If not overwritten in subclass

        """
        raise NotImplementedError

    @abstractmethod
    def _figure(self, *args, **kwargs):
        """
        Abstract Interface Function to log a single ``matplotlib`` figure

        Parameters
        ----------
        *args
            arbitrary positional arguments
        **kwargs
            arbitrary keyword arguments

        Raises
        ------
        NotImplementedError
            If not overwritten in subclass

        """
        raise NotImplementedError

    @abstractmethod
    def _audio(self, *args, **kwargs):
        """
        Abstract Interface Function to log a single audio signal

        Parameters
        ----------
        *args
            arbitrary positional arguments
        **kwargs
            arbitrary keyword arguments

        Raises
        ------
        NotImplementedError
            If not overwritten in subclass

        """
        raise NotImplementedError

    @abstractmethod
    def _video(self, *args, **kwargs):
        """
        Abstract Interface Function to log a single video

        Parameters
        ----------
        *args
            arbitrary positional arguments
        **kwargs
            arbitrary keyword arguments

        Raises
        ------
        NotImplementedError
            If not overwritten in subclass

        """
        raise NotImplementedError

    @abstractmethod
    def _text(self, *args, **kwargs):
        """
        Abstract Interface Function to log a single string as text

        Parameters
        ----------
        *args
            arbitrary positional arguments
        **kwargs
            arbitrary keyword arguments

        Raises
        ------
        NotImplementedError
            If not overwritten in subclass

        """
        raise NotImplementedError

    @abstractmethod
    def _graph_pytorch(self, *args, **kwargs):
        """
        Abstract Interface Function to log a ``PyTorch`` Graph

        Parameters
        ----------
        *args
            arbitrary positional arguments
        **kwargs
            arbitrary keyword arguments

        Raises
        ------
        NotImplementedError
            If not overwritten in subclass

        """

        raise NotImplementedError

    @abstractmethod
    def _graph_tf(self, *args, **kwargs):
        """
        Abstract Interface Function to log a TF Graph

        Parameters
        ----------
        *args
            arbitrary positional arguments
        **kwargs
            arbitrary keyword arguments

        Raises
        ------
        NotImplementedError
            If not overwritten in subclass

        """
        raise NotImplementedError

    @abstractmethod
    def _graph_onnx(self, *args, **kwargs):
        """
        Abstract Interface Function to log a ONNX Graph

        Parameters
        ----------
        *args
            arbitrary positional arguments
        **kwargs
            arbitrary keyword arguments

        Raises
        ------
        NotImplementedError
            If not overwritten in subclass

        """
        raise NotImplementedError

    @abstractmethod
    def _embedding(self, *args, **kwargs):
        """
        Abstract Interface Function to create and log an embedding

        Parameters
        ----------
        *args
            arbitrary positional arguments
        **kwargs
            arbitrary keyword arguments

        Raises
        ------
        NotImplementedError
            If not overwritten in subclass

        """
        raise NotImplementedError

    @abstractmethod
    def _pr_curve(self, *args, **kwargs):
        """
        Abstract Interface Function to calculate and log a PR curve out of
        given values

        Parameters
        ----------
        *args
            arbitrary positional arguments
        **kwargs
            arbitrary keyword arguments

        Raises
        ------
        NotImplementedError
            If not overwritten in subclass

        """
        raise NotImplementedError

    def _scatter(self, plot_kwargs: dict, figure_kwargs: dict = None,
                 **kwargs):
        """
        Function to create a scatter plot and push it

        Parameters
        ----------
        plot_kwargs : dict
            the arguments for plotting
        figure_kwargs : dict
            the arguments to actually create the figure
        **kwargs :
            additional keyword arguments for pushing the created figure to the
            logging writer

        """

        if figure_kwargs is None:
            figure_kwargs = {}
        with self.FigureManager(self._figure, figure_kwargs, kwargs):
            from matplotlib.pyplot import scatter

            scatter(self, **plot_kwargs)

    def _line(self, plot_kwargs=None, figure_kwargs=None, **kwargs):
        """
        Function to create a line plot and push it

        Parameters
        ----------
        plot_kwargs : dict
            the arguments for plotting
        figure_kwargs : dict
            the arguments to actually create the figure
        **kwargs :
            additional keyword arguments for pushing the created figure to the
            logging writer

        """

        if figure_kwargs is None:
            figure_kwargs = {}
        if plot_kwargs is None:
            plot_kwargs = {}
        with self.FigureManager(self._figure, figure_kwargs, kwargs):
            from matplotlib.pyplot import plot
            plot(**plot_kwargs)

    def _stem(self, plot_kwargs=None, figure_kwargs=None, **kwargs):
        """
        Function to create a stem plot and push it

        Parameters
        ----------
        plot_kwargs : dict
            the arguments for plotting
        figure_kwargs : dict
            the arguments to actually create the figure
        **kwargs :
            additional keyword arguments for pushing the created figure to the
            logging writer

        """
        if figure_kwargs is None:
            figure_kwargs = {}
        if plot_kwargs is None:
            plot_kwargs = {}
        with self.FigureManager(self._figure, figure_kwargs, kwargs):
            from matplotlib.pyplot import stem
            stem(**plot_kwargs)

    def _heatmap(self, plot_kwargs=None, figure_kwargs=None, **kwargs):
        """
        Function to create a heatmap plot and push it

        Parameters
        ----------
        plot_kwargs : dict
            the arguments for plotting
        figure_kwargs : dict
            the arguments to actually create the figure
        **kwargs :
            additional keyword arguments for pushing the created figure to the
            logging writer

        """
        if figure_kwargs is None:
            figure_kwargs = {}
        if plot_kwargs is None:
            plot_kwargs = {}
        with self.FigureManager(self._figure, figure_kwargs, kwargs):
            from seaborn import heatmap
            heatmap(**plot_kwargs)

    def _bar(self, plot_kwargs=None, figure_kwargs=None, **kwargs):
        """
        Function to create a bar plot and push it

        Parameters
        ----------
        plot_kwargs : dict
            the arguments for plotting
        figure_kwargs : dict
            the arguments to actually create the figure
        **kwargs :
            additional keyword arguments for pushing the created figure to the
            logging writer

        """
        if figure_kwargs is None:
            figure_kwargs = {}
        if plot_kwargs is None:
            plot_kwargs = {}
        with self.FigureManager(self._figure, figure_kwargs, kwargs):
            from matplotlib.pyplot import bar
            bar(**plot_kwargs)

    def _boxplot(self, plot_kwargs=None, figure_kwargs=None, **kwargs):
        """
        Function to create a boxplot and push it

        Parameters
        ----------
        plot_kwargs : dict
            the arguments for plotting
        figure_kwargs : dict
            the arguments to actually create the figure
        **kwargs :
            additional keyword arguments for pushing the created figure to the
            logging writer

        """
        if plot_kwargs is None:
            plot_kwargs = {}
        if figure_kwargs is None:
            figure_kwargs = {}
        with self.FigureManager(self._figure, figure_kwargs, kwargs):
            from matplotlib.pyplot import boxplot
            boxplot(**plot_kwargs)

    def _surface(self, plot_kwargs=None, figure_kwargs=None, **kwargs):
        """
        Function to create a surface plot and push it

        Parameters
        ----------
        plot_kwargs : dict
            the arguments for plotting
        figure_kwargs : dict
            the arguments to actually create the figure
        **kwargs :
            additional keyword arguments for pushing the created figure to the
            logging writer

        """
        if figure_kwargs is None:
            figure_kwargs = {}
        if plot_kwargs is None:
            plot_kwargs = {}
        with self.FigureManager(self._figure, figure_kwargs, kwargs):
            from seaborn import kdeplot

            kdeplot(**plot_kwargs)

    def _contour(self, plot_kwargs=None, figure_kwargs=None, **kwargs):
        """
        Function to create a contour plot and push it

        Parameters
        ----------
        plot_kwargs : dict
            the arguments for plotting
        figure_kwargs : dict
            the arguments to actually create the figure
        **kwargs :
            additional keyword arguments for pushing the created figure to the
            logging writer

        """
        if figure_kwargs is None:
            figure_kwargs = {}
        if plot_kwargs is None:
            plot_kwargs = {}
        with self.FigureManager(self._figure, figure_kwargs, kwargs):
            from matplotlib.pyplot import contour

            contour(**plot_kwargs)

    def _quiver(self, plot_kwargs=None, figure_kwargs=None, **kwargs):
        """
        Function to create a quiver plot and push it

        Parameters
        ----------
        plot_kwargs : dict
            the arguments for plotting
        figure_kwargs : dict
            the arguments to actually create the figure
        **kwargs :
            additional keyword arguments for pushing the created figure to the
            logging writer

        """
        if plot_kwargs is None:
            plot_kwargs = {}
        if figure_kwargs is None:
            figure_kwargs = {}
        with self.FigureManager(self._figure, figure_kwargs, kwargs):
            from matplotlib.pyplot import quiver
            quiver(**plot_kwargs)

    @property
    def name(self):
        return "BaseBackend"
