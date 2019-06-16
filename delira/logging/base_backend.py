
from queue import Empty
from abc import abstractmethod, ABCMeta
from threading import Event
from queue import Queue

_FUNCTIONS_WITHOUT_STEP = ["graph_pytorch", "graph_tf", "graph_onnx",
                           "embedding"]


class BaseBackend(object, metaclass=ABCMeta):

    class FigureManager:

        def __init__(self, push_fn, figure_kwargs: dict, push_kwargs: dict):
            self._push_fn = push_fn
            self._figure_kwargs = figure_kwargs
            self._push_kwargs = push_kwargs
            self._fig = None

        def __enter__(self):
            from matplotlib.pyplot import figure
            self._fig = figure(**self._figure_kwargs)

        def __exit__(self, *args):
            from matplotlib.pyplot import close
            self._push_fn(figure=self._fig, **self._push_kwargs)

            close(self._fig)
            self._fig = None

    def __init__(self, abort_event: Event = None, queue: Queue = None):
        super().__init__()
        self.KEYWORD_FN_MAPPING = {}

        self.daemon = True

        self._queue = queue
        self._abort_event = abort_event
        self._global_steps = {}

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
        process_item = self._queue.get_nowait()
        if isinstance(process_item, dict):
            for key, val in process_item.items():
                execute_fn = self.KEYWORD_FN_MAPPING[str(key).lower()]
                val = self.resolve_global_step(str(key).lower(), **val)

                self._call_exec_fn(execute_fn, val)

        else:
            raise ValueError("Invalid Value passed for logging: %s"
                             % str(process_item))

    def resolve_global_step(self, key, **val):
        # check if function should be processed statically
        # (no time update possible)
        if str(key).lower() not in _FUNCTIONS_WITHOUT_STEP:

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
                    step = self._global_steps[val[tag]]
                    self._global_steps[val[tag]] += 1

                else:
                    # if not existent_ set step for given tag to zero
                    step = 0
                    self._global_steps[val[tag]] = step

                val.update({"global_step": step})

            elif "global_step" in val:
                self._global_steps[tag] = val["global_step"]

        return val

    def run(self):
        try:
            self._log_item()

        except Empty:
            pass

        except Exception as e:
            self._abort_event.set()
            raise e

    def set_queue(self, queue: Queue):
        self._queue = queue

    def set_event(self, event: Event):
        self._abort_event = event

    def _call_exec_fn(self, exec_fn, args):

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
        raise NotImplementedError

    @abstractmethod
    def _images(self, *args, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def _image_with_boxes(self, *args, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def _scalar(self, *args, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def _scalars(self, *args, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def _histogram(self, *args, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def _figure(self, *args, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def _audio(self, *args, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def _video(self, *args, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def _text(self, *args, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def _graph_pytorch(self, *args, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def _graph_tf(self, *args, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def _graph_onnx(self, *args, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def _embedding(self, *args, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def _pr_curve(self, *args, **kwargs):
        raise NotImplementedError

    def _scatter(self, plot_kwargs: dict, figure_kwargs={}, **kwargs):

        with self.FigureManager(self._figure, figure_kwargs, kwargs):
            from matplotlib.pyplot import scatter

            scatter(self, **plot_kwargs)

    def _line(self, plot_kwargs={}, figure_kwargs={}, **kwargs):

        with self.FigureManager(self._figure, figure_kwargs, kwargs):
            from matplotlib.pyplot import plot
            plot(**plot_kwargs)

    def _stem(self, plot_kwargs={}, figure_kwargs={}, **kwargs):
        with self.FigureManager(self._figure, figure_kwargs, kwargs):
            from matplotlib.pyplot import stem
            stem(**plot_kwargs)

    def _heatmap(self, plot_kwargs={}, figure_kwargs={}, **kwargs):
        with self.FigureManager(self._figure, figure_kwargs, kwargs):
            from seaborn import heatmap
            heatmap(**plot_kwargs)

    def _bar(self, plot_kwargs={}, figure_kwargs={}, **kwargs):
        with self.FigureManager(self._figure, figure_kwargs, kwargs):
            from matplotlib.pyplot import bar
            bar(**plot_kwargs)

    def _boxplot(self, plot_kwargs={}, figure_kwargs={}, **kwargs):
        with self.FigureManager(self._figure, figure_kwargs, kwargs):
            from matplotlib.pyplot import boxplot
            boxplot(**plot_kwargs)

    def _surface(self, plot_kwargs={}, figure_kwargs={}, **kwargs):
        with self.FigureManager(self._figure, figure_kwargs, kwargs):
            from seaborn import kdeplot

            kdeplot(**plot_kwargs)

    def _contour(self, plot_kwargs={}, figure_kwargs={}, **kwargs):
        with self.FigureManager(self._figure, figure_kwargs, kwargs):
            from matplotlib.pyplot import contour

            contour(**plot_kwargs)

    def _quiver(self, plot_kwargs={}, figure_kwargs={}, **kwargs):
        with self.FigureManager(self._figure, figure_kwargs, kwargs):
            from matplotlib.pyplot import quiver
            quiver(**plot_kwargs)
