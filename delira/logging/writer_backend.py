
from delira.logging.base_backend import BaseBackend
from queue import Queue
from threading import Event


class WriterLoggingBackend(BaseBackend):
    """
    A Basic Writer Backend for a unspecified writer class
    """

    def __init__(self, writer_cls, writer_kwargs: dict,
                 abort_event: Event = None, queue: Queue = None):
        super().__init__(abort_event, queue)

        self._writer = writer_cls(**writer_kwargs)

    @staticmethod
    def convert_to_npy(*args, **kwargs):
        """
        Function to convert all positional args and keyword args to numpy
        (returns identity per default, but can be overwritten in subclass to
        log more complex types)

        Parameters
        ----------
        *args :
            positional arguments of arbitrary number and type
        **kwargs :
            keyword arguments of arbitrary number and type
        Returns
        -------
        tuple
            converted positional arguments
        dict
            converted keyword arguments
        """
        return args, kwargs

    def _image(self, tag, img_tensor, global_step=None, walltime=None,
               dataformats='CHW'):
        """
        Function to log a single image

        Parameters
        ----------
        tag : str
            the tag to store the image at
        img_tensor : array-like
            an array-like object containing the actual image; Must be
            convertible to numpy
        global_step : int
            the global step
        walltime :
            the overall time
        dataformats : str
            string specifying the image format

        """
        converted_args, converted_kwargs = self.convert_to_npy(
            tag=tag, img_tensor=img_tensor, global_step=global_step,
            walltime=walltime, dataformats=dataformats)

        self._writer.add_image(*converted_args, **converted_kwargs)

    def _images(self, tag, img_tensor, global_step=None, walltime=None,
                dataformats='NCHW'):
        """
        Function to log multiple values

        Parameters
        ----------
        tag : str
            the tag to store the image at
        img_tensor : array-like
            an array-like object containing the actual image; Must be
            convertible to numpy
        global_step : int
            the global step
        walltime :
            the overall time
        dataformats : str
            string specifying the image format

        """
        converted_args, converted_kwargs = self.convert_to_npy(
            tag=tag, img_tensor=img_tensor, global_step=global_step,
            walltime=walltime, dataformats=dataformats)

        self._writer.add_images(*converted_args, **converted_kwargs)

    def _image_with_boxes(self, tag, img_tensor, box_tensor, global_step=None,
                          walltime=None, dataformats='CHW', **kwargs):
        """
        Function to log a single image with bounding boxes

        Parameters
        ----------
        tag : str
            the tag to store the image at
        img_tensor : array-like
            an array-like object containing the actual image; Must be
            convertible to numpy
        box_tensor : array-like
            an array-like object containing the actual bounding boxes in xyxy
            format; must be convertible to numpy
        global_step : int
            the global step
        walltime :
            the overall time
        dataformats : str
            string specifying the image format
        **kwargs :
            additional keyword arguments

        """
        converted_args, converted_kwargs = self.convert_to_npy(
            tag=tag, img_tensor=img_tensor, box_tensor=box_tensor,
            global_step=global_step, walltime=walltime,
            dataformats=dataformats, **kwargs)

        self._writer.add_image_with_boxes(*converted_args, **converted_kwargs)

    def _scalar(self, tag, scalar_value, global_step=None, walltime=None):
        """
        Function to log a single scalar value

        Parameters
        ----------
        tag : str
            the tag to store the image at
        scalar_value : int or float
            the scalar value to log
        global_step : int
            the global step
        walltime :
            the overall time

        """
        converted_args, converted_kwargs = self.convert_to_npy(
            tag=tag, scalar_value=scalar_value, global_step=global_step,
            walltime=walltime)
        self._writer.add_scalar(*converted_args, **converted_kwargs)

    def _scalars(self, main_tag, tag_scalar_dict, global_step=None,
                 walltime=None):
        """
        Function to log multiple scalars

        Parameters
        ----------
        main_tag : str
            the main tag to store the scalars at
        tag_scalar_dict : dict
            a dictionary containing tags as keys and the corresponding scalar
            values
        global_step : int
            the global step
        walltime :
            the overall time

        """
        converted_args, converted_kwargs = self.convert_to_npy(
            main_tag=main_tag, tag_scalar_dict=tag_scalar_dict,
            global_step=global_step, walltime=walltime)

        self._writer.add_scalars(*converted_args, **converted_kwargs)

    def _histogram(self, tag, values, global_step=None, bins='tensorflow',
                   walltime=None):
        """
        Function to create and log a histogram out of given values

        Parameters
        ----------
        tag : str
            the tag to store the histogram at
        values : arraylike
            an arraylike object containing the raw data to create a histogram
            from; Must be convertible to numpy
        global_step : int
            global step
        bins : str
            string indicating the bins format
        walltime :
            the overall time


        """
        converted_args, converted_kwargs = self.convert_to_npy(
            tag=tag, values=values, global_step=global_step, bins=bins)
        self._writer.add_histogram(*converted_args, **converted_kwargs)

    def _figure(self, tag, figure, global_step=None, close=True,
                walltime=None):
        """
        Function to log a ``matplotlib.pyplot`` figure

        Parameters
        ----------
        tag : str
            the tag to store the figure at
        figure : :class:`matplotlib.pyplot.Figure``
            the figure to log
        global_step : int
            the global step
        close : bool
            whether to close the figure after pushing it
        walltime :
            the overall time

        """
        converted_args, converted_kwargs = self.convert_to_npy(
            tag=tag, figure=figure, global_step=global_step, close=close,
            walltime=walltime)
        self._writer.add_figure(*converted_args, **converted_kwargs)

    def _audio(self, tag, snd_tensor, global_step=None, sample_rate=44100,
               walltime=None):
        """
        Function to log a single audio signal
        Parameters
        ----------
        tag : str
            the tag to store the sound signal at
        snd_tensor : arraylike
            arraylike object containing the sound signal;
            must be convertible to numpy
        global_step : int
            the global step
        sample_rate : int
            the sampling rate for the sound signal
        walltime :
            the overall time

        """
        converted_args, converted_kwargs = self.convert_to_npy(
            tag=tag, snd_tensor=snd_tensor, global_step=global_step,
            sample_rate=sample_rate, walltime=walltime
        )
        self._writer.add_audio(*converted_args, **converted_kwargs)

    def _text(self, tag, text_string, global_step=None, walltime=None):
        """
        Function to log a single string as text

        Parameters
        ----------
        tag : str
            the tag to store the text at
        text_string : str
            the text string to log
        global_step : int
            the global step
        walltime :
            the overall time

        """
        converted_args, converted_kwargs = self.convert_to_npy(
            tag=tag, text_string=text_string, global_step=global_step,
            walltime=walltime)
        self._writer.add_text(*converted_args, **converted_kwargs)

    def _pr_curve(self, tag, labels, predictions, global_step=None,
                  num_thresholds=127, weights=None, walltime=None):
        """
        Function to create and log a PR curve out of given predictions and +
        labels

        Parameters
        ----------
        tag : str
            function to store the curve at
        labels : arraylike
            arraylike object containing the groundtruth data; must be
            convertible to numpy
        predictions : arraylike
            arraylike object containing the predictions; must be convertible
            to numpy
        global_step : int
            the global step
        num_thresholds : int
            number of thresholds to apply for PR calculation
        weights : arraylike
            arraylike object containing sample weights, must be covertible to
            numpy
        walltime :
            overall time

        """
        converted_args, converted_kwargs = self.convert_to_npy(
            tag=tag, labels=labels, predictions=predictions,
            global_step=global_step, num_thresholds=num_thresholds,
            weights=weights, walltime=walltime)
        self._writer.add_pr_curve(*converted_args, **converted_kwargs)

    def _video(self, tag, vid_tensor, global_step=None, fps=4, walltime=None):
        """
        Function to log a single video

        Parameters
        ----------
        tag : str
            the tag to store the image at
        vid_tensor : arraylike
            arraylike object containing the video frames; must be convertible
            to numpy
        global_step : int
            the global step
        fps : int
            frames per second to display
        walltime : int
            the overall time

        """
        converted_args, converted_kwargs = self.convert_to_npy(
            tag=tag, vid_tensor=vid_tensor, global_step=global_step, fps=fps,
            walltime=walltime)
        self._writer.add_video(*converted_args, **converted_kwargs)

    @property
    def name(self):
        return "WriterBackend"
