import tensorboardX
from threading import Event
from queue import Queue

from delira.logging.writer_backend import WriterLoggingBackend


class TensorboardBackend(WriterLoggingBackend):
    def __init__(self, writer_kwargs: dict = {},
                 abort_event: Event = None, queue: Queue = None):

        super().__init__(tensorboardX.SummaryWriter, writer_kwargs,
                         abort_event, queue)

    def _call_exec_fn(self, exec_fn, args):
        ret_val = super()._call_exec_fn(exec_fn, args)

        self._writer.file_writer.flush()

        return ret_val
