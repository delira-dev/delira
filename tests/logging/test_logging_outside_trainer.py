import unittest
from delira.logging import log
from delira.training import BaseNetworkTrainer
from delira.training.callbacks import DefaultLoggingCallback
from delira.models import AbstractNetwork
import logging
import os


class DummyNetwork(AbstractNetwork):
    def __call__(self, *args, **kwargs):
        return {}

    @staticmethod
    def closure(model, data_dict: dict, optimizers: dict, losses: dict,
                iter_num: int, fold=0, **kwargs):
        return losses, data_dict

    @staticmethod
    def prepare_batch(batch: dict, input_device, output_device):
        return batch


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
            logging_kwargs={})

        trainer._setup(
            AbstractNetwork(),
            lr_scheduler_cls=None,
            lr_scheduler_params={},
            gpu_ids=[],
            key_mapping={},
            convert_batch_to_npy_fn=None,
            prepare_batch_fn=None,
            callbacks=[])


if __name__ == '__main__':
    unittest.main()
