
from delira.logging.base_backend import BaseBackend
from queue import Queue
from threading import Event


class WriterLoggingBackend(BaseBackend):
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

    def _image(self,  tag, img_tensor, global_step=None, walltime=None,
               dataformats='CHW'):
        converted_args, converted_kwargs = self.convert_to_npy(
            tag=tag, img_tensor=img_tensor, global_step=global_step,
            walltime=walltime, dataformats=dataformats)

        self._writer.add_image(*converted_args, **converted_kwargs)

    def _images(self,  tag, img_tensor, global_step=None, walltime=None,
                dataformats='NCHW'):
        converted_args, converted_kwargs = self.convert_to_npy(
            tag=tag, img_tensor=img_tensor, global_step=global_step,
            walltime=walltime, dataformats=dataformats)

        self._writer.add_images(*converted_args, **converted_kwargs)

    def _image_with_boxes(self, tag, img_tensor, box_tensor, global_step=None,
                          walltime=None, dataformats='CHW', **kwargs):
        """xyxy format for boxes"""
        converted_args, converted_kwargs = self.convert_to_npy(
            tag=tag, img_tensor=img_tensor, box_tensor=box_tensor,
            global_step=global_step, walltime=walltime,
            dataformats=dataformats, **kwargs)

        self._writer.add_image_with_boxes(*converted_args, **converted_kwargs)

    def _scalar(self, tag, scalar_value, global_step=None, walltime=None):
        converted_args, converted_kwargs = self.convert_to_npy(
            tag=tag, scalar_value=scalar_value, global_step=global_step,
            walltime=walltime)
        self._writer.add_scalar(*converted_args, **converted_kwargs)

    def _scalars(self, main_tag, tag_scalar_dict, global_step=None,
                 walltime=None):
        converted_args, converted_kwargs = self.convert_to_npy(
            main_tag=main_tag, tag_scalar_dict=tag_scalar_dict,
            global_step=global_step, walltime=walltime)

        self._writer.add_scalars(*converted_args, **converted_kwargs)

    def _histogram(self, tag, values, global_step=None, bins='tensorflow',
                   walltime=None):
        converted_args, converted_kwargs = self.convert_to_npy(
            tag=tag, values=values, global_step=global_step, bins=bins)
        self._writer.add_histogram(*converted_args, **converted_kwargs)

    def _figure(self, tag, figure, global_step=None, close=True, walltime=None):
        converted_args, converted_kwargs = self.convert_to_npy(
            tag=tag, figure=figure, global_step=global_step, close=close,
            walltime=walltime)
        self._writer.add_figure(*converted_args, **converted_kwargs)

    def _audio(self, tag, snd_tensor, global_step=None, sample_rate=44100, walltime=None):
        converted_args, converted_kwargs = self.convert_to_npy(
            tag=tag, snd_tensor=snd_tensor, global_step=global_step,
            sample_rate=sample_rate, walltime=walltime
        )
        self._writer.add_audio(*converted_args, **converted_kwargs)

    def _text(self, tag, text_string, global_step=None, walltime=None):
        converted_args, converted_kwargs = self.convert_to_npy(
            tag=tag, text_string=text_string, global_step=global_step,
            walltime=walltime)
        self._writer.add_text(*converted_args, **converted_kwargs)

    def _graph_pytorch(self, model, input_to_model=None, verbose=False, **kwargs):
        converted_args, converted_kwargs = self.convert_to_npy(
            model=model, input_to_model=input_to_model,
            verbose=verbose, **kwargs)

        self._writer.add_graph(*converted_args, **converted_kwargs)

    def _graph_tf(self, graph):
        import tensorflow as tf
        from tensorboardX.proto.event_pb2 import Event

        if isinstance(graph, tf.Graph):
            graphdef = graph.as_graph_def()
        elif isinstance(graph, tf.GraphDef):
            graphdef = graph
        elif hasattr(graph, "SerializeToString"):
            graphdef = graph
        else:
            raise TypeError("Invalid type given for graph: %s" %
                            graph.__class__.__name__)

        self._writer.add_event(Event(graph_def=graphdef.SerializeToString()))

    def _graph_onnx(self, prototxt):
        converted_args, converted_kwargs = self.convert_to_npy(
            prototxt=prototxt)
        self._writer.add_onnx_graph(*converted_args, **converted_kwargs)

    def _embedding(self, mat, metadata=None, label_img=None, global_step=None,
                   tag='default', metadata_header=None):
        converted_args, converted_kwargs = self.convert_to_npy(
            mat=mat, metadata=metadata, label_img=label_img,
            global_step=global_step
        )
        self._writer.add_embedding(*converted_args, **converted_kwargs)

    def _pr_curve(self, tag, labels, predictions, global_step=None,
                  num_thresholds=127, weights=None, walltime=None):
        converted_args, converted_kwargs = self.convert_to_npy(
            tag=tag, labels=labels, predictions=predictions,
            global_step=global_step, num_thresholds=num_thresholds,
            weights=weights, walltime=walltime)
        self._writer.add_pr_curve(*converted_args, **converted_kwargs)

    def _video(self, tag, vid_tensor, global_step=None, fps=4, walltime=None):
        converted_args, converted_kwargs = self.convert_to_npy(
            tag=tag, vid_tensor=vid_tensor, global_step=global_step, fps=fps,
            walltime=walltime)
        self._writer.add_video(*converted_args, **converted_kwargs)
