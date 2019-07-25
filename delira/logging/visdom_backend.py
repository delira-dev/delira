import tensorboardX
from threading import Event
from queue import Queue

from delira.logging.writer_backend import WriterLoggingBackend


class VisdomBackend(WriterLoggingBackend):
    """
    A Visdom Logging backend
    """

    def __init__(self, writer_kwargs: dict = None,
                 abort_event: Event = None, queue: Queue = None):
        """

        Parameters
        ----------
        writer_kwargs : dict
            arguments to initialize a writer
        abort_event : :class:`threading.Event`
            the abortion event
        queue : :class:`queue.Queue`
            the queue holding all logging tasks
        """

        if writer_kwargs is None:
            writer_kwargs = {}

        super().__init__(
            tensorboardX.visdom_writer.VisdomWriter,
            writer_kwargs,
            abort_event,
            queue)

    @property
    def name(self):
        return "VisdomBackend"
