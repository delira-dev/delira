import os
import delira
import unittest

import numpy as np


class TfExperimentTest(unittest.TestCase):

    @unittest.skipIf("TF" not in delira.get_backends(),
                     reason="No TF Backend installed")
    def test_experiment(self):

        from delira.training import TfExperiment, Parameters
        from delira.models.classification import ClassificationNetworkBaseTf
        from delira.data_loading import AbstractDataset, BaseDataManager
        import tensorflow as tf

        test_cases = [
            (
                Parameters(fixed_params={
                    "model": {'in_channels': 32,
                              'n_outputs': 1},
                    "training": {
                        "criterions": {"CE":
                                       tf.losses.softmax_cross_entropy},
                        "optimizer_cls": tf.train.AdamOptimizer,
                        "optimizer_params": {"learning_rate": 1e-3},
                        "num_epochs": 2,
                        "metrics": {},
                        "lr_sched_cls": None,
                        "lr_sched_params": {}}
                }
                ),
                500,
                50)
        ]

        class DummyNetwork(ClassificationNetworkBaseTf):
            def __init__(self):
                super().__init__(32, 1)
                self.model = self._build_model(1)

                images = tf.placeholder(shape=[None, 32],
                                        dtype=tf.float32)
                labels = tf.placeholder(shape=[None, 1], dtype=tf.float32)

                preds_train = self.model(images, training=True)
                preds_eval = self.model(images, training=False)

                self.inputs = [images, labels]
                self.outputs_train = [preds_train]
                self.outputs_eval = [preds_eval]

            @staticmethod
            def _build_model(n_outputs):
                return tf.keras.models.Sequential(
                    layers=[
                        tf.keras.layers.Dense(64, input_shape=(
                            32,), bias_initializer='glorot_uniform'),
                        tf.keras.layers.ReLU(),
                        tf.keras.layers.Dense(n_outputs, bias_initializer='glorot_uniform')]
                )

        class DummyDataset(AbstractDataset):
            def __init__(self, length):
                super().__init__(None, None, None, None)
                self.length = length

            def __getitem__(self, index):
                return {"data": np.random.rand(32),
                        "label": np.random.randint(0, 1, 1)}

            def __len__(self):
                return self.length

            def get_sample_from_index(self, index):
                return self.__getitem__(index)

        for case in test_cases:
            with self.subTest(case=case):

                params, dataset_length_train, dataset_length_test = case

                exp = TfExperiment(params, DummyNetwork)
                dset_train = DummyDataset(dataset_length_train)
                dset_test = DummyDataset(dataset_length_test)

                dmgr_train = BaseDataManager(dset_train, 16, 4, None)
                dmgr_test = BaseDataManager(dset_test, 16, 1, None)

                net = exp.run(dmgr_train, dmgr_test)
                exp.test(params=params,
                         network=net,
                         datamgr_test=dmgr_test, )

                exp.kfold(2, dmgr_train, num_splits=2)
                exp.stratified_kfold(2, dmgr_train, num_splits=2)
                exp.stratified_kfold_predict(2, dmgr_train, num_splits=2)


if __name__ == '__main__':
    unittest.main()
