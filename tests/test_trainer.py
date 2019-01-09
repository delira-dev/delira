import pytest

import numpy as np

from delira.training import PyTorchNetworkTrainer
from delira.models.classification import ClassificationNetworkBasePyTorch
from delira.data_loading import AbstractDataset, BaseDataManager
import torch


@pytest.mark.parametrize("criterions,optimizer_cls,optimizer_params,"
                         "dataset_length_train,dataset_length_test",
                         [
                             ({"CE": torch.nn.CrossEntropyLoss()},
                              torch.optim.Adam,
                              {"lr": 1e-3},
                              500,
                              50)
                         ])
def test_trainer(criterions, optimizer_cls, optimizer_params,
                 dataset_length_train, dataset_length_test):
    class DummyNetwork(ClassificationNetworkBasePyTorch):
        def __init__(self):
            super().__init__(32, 1)

        def forward(self, x):
            return self.module(x)

        @staticmethod
        def _build_model(in_channels, n_outputs):
            return torch.nn.Sequential(
                torch.nn.Linear(in_channels, 64),
                torch.nn.ReLU(),
                torch.nn.Linear(64, n_outputs)
            )

    class DummyDataset(AbstractDataset):
        def __init__(self, length):
            super().__init__(None, None, None, None)
            self.length = length

        def __getitem__(self, index):
            return {"data": np.random.rand(1, 32),
                    "label": np.random.rand(1, 1)}

        def __len__(self):
            return self.length

    network = DummyNetwork()
    trainer = PyTorchNetworkTrainer(
        network=network, save_path="/tmp/delira_trainer_test",
        criterions=criterions, optimizer_cls=optimizer_cls,
        optimizer_params=optimizer_params)

    dset_train = DummyDataset(dataset_length_train)
    dset_test = DummyDataset(dataset_length_test)

    dmgr_train = BaseDataManager(dset_train, 16, 4, None)
    dmgr_test = BaseDataManager(dset_test, 16, 1, None)

    trainer.train(2, dmgr_train, dmgr_test)

if __name__ == '__main__':
    test_trainer()