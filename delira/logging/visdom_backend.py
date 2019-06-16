import tensorboardX
from threading import Event
from queue import Queue

from delira.logging.writer_backend import WriterLoggingBackend


class VisdomBackend(WriterLoggingBackend):
    def __init__(self, writer_kwargs: dict = {},
                 abort_event: Event = None, queue: Queue = None):
        super().__init__(tensorboardX.visdom_writer.VisdomWriter, writer_kwargs,
                         abort_event, queue)
