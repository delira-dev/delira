from delira import get_backends
import unittest
import numpy as np
from functools import partial

from delira.data_loading import AbstractDataset, BaseDataManager


class DummyDataset(AbstractDataset):
    def __init__(self, length):
        super().__init__(None, None)
        self.length = length

    def __getitem__(self, index):
        return {"data": np.random.rand(32),
                "label": np.random.randint(0, 1, 1)}

    def __len__(self):
        return self.length

    def get_sample_from_index(self, index):
        return self.__getitem__(index)


class TestPredictor(unittest.TestCase):

    def setUp(self):

        self.dset = DummyDataset(20)
        self.dmgr = BaseDataManager(self.dset, 4, 1, transforms=None)

    # @unittest.skip
    @unittest.skipIf("TF" not in get_backends(),
                     reason="No TF Backend installed")
    def test_predictor_tf(self):
        from delira.training import Predictor
        from delira.models.classification import ClassificationNetworkBaseTf
        from delira.models import AbstractTfNetwork
        from delira.training.train_utils import convert_tf_tensor_to_npy,\
            initialize_uninitialized

        import tensorflow as tf

        class DummyNetwork(ClassificationNetworkBaseTf):
            def __init__(self):
                AbstractTfNetwork.__init__(self)
                self.model = self._build_model(1)

                images = tf.placeholder(shape=[None, 32],
                                        dtype=tf.float32)
                labels = tf.placeholder_with_default(
                    tf.zeros(
                        [tf.shape(images)[0], 1]),
                    shape=[None, 1])

                preds_train = self.model(images, training=True)
                preds_eval = self.model(images, training=False)

                self.inputs["images"] = images
                self.inputs["labels"] = labels
                self.outputs_train["pred"] = preds_train
                self.outputs_eval["pred"] = preds_eval

            @staticmethod
            def _build_model(n_outputs):
                return tf.keras.models.Sequential(
                    layers=[
                        tf.keras.layers.Dense(
                            64,
                            input_shape=(
                                32,
                            ),
                            bias_initializer='glorot_uniform'),
                        tf.keras.layers.ReLU(),
                        tf.keras.layers.Dense(
                            n_outputs,
                            bias_initializer='glorot_uniform')])

        net = DummyNetwork()
        initialize_uninitialized(net._sess)

        predictor = Predictor(
            net,
            {"images": "data"},
            # prepare_batch_fn=partial(DummyNetwork.prepare_batch,
            #                                          input_device="cpu",
            #                                          output_device="cpu"),
            convert_batch_to_npy_fn=convert_tf_tensor_to_npy
        )

        # add singleton dimension for test case
        batch = self.dset[0]
        batch['data'] = batch['data'][np.newaxis, :]

        pred = predictor.predict(batch)

        pred_dmgr = predictor.predict_data_mgr(self.dmgr)

    @unittest.skipIf("TORCH" not in get_backends(),
                     reason="No Torch Backend Installed")
    def test_predictor_torch(self):
        from delira.training import Predictor
        from delira.models import ClassificationNetworkBasePyTorch
        from delira.training.train_utils import convert_torch_tensor_to_npy

        import torch

        class DummyNetwork(ClassificationNetworkBasePyTorch):

            def __init__(self):
                super().__init__(32, 1)

            def forward(self, x):
                return {"pred": self.module(x)}

            @staticmethod
            def _build_model(in_channels, n_outputs):
                return torch.nn.Sequential(
                    torch.nn.Linear(in_channels, 64),
                    torch.nn.ReLU(),
                    torch.nn.Linear(64, n_outputs)
                )

            @staticmethod
            def prepare_batch(batch_dict, input_device, output_device):
                return {"data": torch.from_numpy(batch_dict["data"]
                                                 ).to(input_device,
                                                      torch.float),
                        "label": torch.from_numpy(batch_dict["label"]
                                                  ).to(output_device,
                                                       torch.long)}

        predictor = Predictor(
            DummyNetwork(),
            {"x": "data"},
            convert_torch_tensor_to_npy,
            prepare_batch_fn=partial(DummyNetwork.prepare_batch,
                                     input_device="cpu", output_device="cpu")
        )

        pred = predictor.predict(self.dset[0])

        # print(pred)
        pred_dmgr = predictor.predict_data_mgr(self.dmgr)
        # print(pred_dmgr)


if __name__ == '__main__':
    from multiprocessing import freeze_support
    freeze_support()
    unittest.main()
