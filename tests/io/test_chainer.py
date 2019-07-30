import unittest

from ..utils import check_for_chainer_backend

if check_for_chainer_backend():
    import chainer
    from delira.models import AbstractChainerNetwork

    # define model outside actual test to make it pickleable
    class Model(AbstractChainerNetwork):
        def __init__(self):
            super().__init__()

            with self.init_scope():
                self.dense = chainer.links.Linear(1, 1)

        def forward(self, x):
            return {
                "pred":
                    chainer.functions.relu(
                        self.dense(x))
            }


class IoChainerTest(unittest.TestCase):

    @unittest.skipUnless(check_for_chainer_backend(),
                         "Test should be only executed if chainer backend is "
                         "installed and specified")
    def test_load_save(self):

        from delira.io.chainer import load_checkpoint, save_checkpoint

        net = Model()

        save_checkpoint("./model_chainer.chain", model=net)
        self.assertTrue(load_checkpoint("./model_chainer.chain", model=net))


if __name__ == '__main__':
    unittest.main()
