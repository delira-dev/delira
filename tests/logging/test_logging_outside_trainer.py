import unittest
from delira.logging import log
from delira.training import BaseNetworkTrainer
from delira.models import AbstractNetwork
import os
import tensorflow as tf


class LoggingOutsideTrainerTestCase(unittest.TestCase):

    def test_logging_freq(self):
        save_path = os.path.abspath("./logs")
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
                'logdir': save_path
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

        tag = 'dummy'

        log({"scalar": {"scalar_value": 1234, "tag": tag}})

        file = [os.path.join(save_path, x)
                for x in os.listdir(save_path)
                if os.path.isfile(os.path.join(save_path, x))][0]

        ret_val = False
        if tf is not None:
            for e in tf.train.summary_iterator(file):
                for v in e.summary.value:
                    if v.tag == tag:
                        ret_val = True
                        break
                if ret_val:
                    break

        self.assertTrue(ret_val)


if __name__ == '__main__':
    unittest.main()
