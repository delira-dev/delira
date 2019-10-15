import unittest
from delira.logging import BaseBackend, SingleThreadedLogger
import logging


class DummyBackend(BaseBackend):
    def _text(self, logging_no: int, tag: str, global_step=None):
        logging.info("INFO: Logging Item Number %d" % logging_no)

    # implement dummy funtions to be able to instantiate backend
    def _image(self, *args, **kwargs):
        pass

    def _images(self, *args, **kwargs):
        pass

    def _image_with_boxes(self, *args, **kwargs):
        pass

    def _scalar(self, *args, **kwargs):
        pass

    def _scalars(self, *args, **kwargs):
        pass

    def _histogram(self, *args, **kwargs):
        pass

    def _figure(self, *args, **kwargs):
        pass

    def _audio(self, *args, **kwargs):
        pass

    def _video(self, *args, **kwargs):
        pass

    def _graph_pytorch(self, *args, **kwargs):
        pass

    def _graph_tf(self, *args, **kwargs):
        pass

    def _graph_onnx(self, *args, **kwargs):
        pass

    def _embedding(self, *args, **kwargs):
        pass

    def _pr_curve(self, *args, **kwargs):
        pass


class LoggingFrequencyTestCase(unittest.TestCase):

    def _logging_freq_test(self, frequencies, num_runs: int, check_freq=None):
        logger = SingleThreadedLogger(DummyBackend(),
                                      logging_frequencies=frequencies,
                                      reduce_types="last")

        if check_freq is None and isinstance(frequencies, int):
            check_freq = frequencies

        assert check_freq is not None

        target_messages = 0

        with self.assertLogs() as cm:
            for idx in range(num_runs):
                logger.log({"text": {"logging_no": idx, "tag": "dummy"}})

                target_messages += int((idx + 1) % check_freq == 0)

        self.assertIsNotNone(cm.output)
        self.assertEqual(target_messages, len(cm.output))

    def test_logging_freq(self):
        for frequencies, check_freq in zip([1, 5, 10, {"text": 15}],
                                           [None, None, None, 15]):
            with self.subTest(frequencies=frequencies):
                self._logging_freq_test(frequencies, 50, check_freq)


if __name__ == '__main__':
    unittest.main()
