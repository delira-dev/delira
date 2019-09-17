import unittest
from delira.logging import log
from delira.training import BaseNetworkTrainer
from delira.training.callbacks import DefaultLoggingCallback
from delira.models import AbstractNetwork
import logging
import os
import logging


class LoggingOutsideTrainerTestCase(unittest.TestCase):

    def test_logging_freq(self):
        save_path = os.path.abspath(".")
        config = {
            "num_epochs": 2,
            "losses": {},
            "optimizer_cls": None,
            "optimizer_params": {"learning_rate": 1e-3},
            "metrics": {},
            "lr_scheduler_cls": None,
            "lr_scheduler_params": {}
        }
        trainer = BaseNetworkTrainer(
            AbstractNetwork(),
            save_path,
            **config,
            gpu_ids=[],
            save_freq=1,
            optim_fn=None,
            key_mapping={},
            logging_type="tensorboardx",
            logging_kwargs={
                "level": logging.INFO
            })

        trainer._setup(
            AbstractNetwork(),
            lr_scheduler_cls=None,
            lr_scheduler_params={},
            gpu_ids=[],
            key_mapping={},
            convert_batch_to_npy_fn=None,
            prepare_batch_fn=None,
            callbacks=[])

        with self.assertLogs() as cm:
            log({"text": {"text_string": "Logging outside trainer", "tag": "dummy"}})

        print(len(cm.output))


if __name__ == '__main__':
    unittest.main()
