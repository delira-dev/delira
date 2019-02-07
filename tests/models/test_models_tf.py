import os
if "tf" in os.environ["DELIRA_BACKEND"]:
    from delira.models import ClassificationNetworkBaseTf
    from delira.training.train_utils import create_optims_default_tf
    import numpy as np
    import pytest
    import time
    import gc
    import sys
    import tensorflow as tf
    from psutil import virtual_memory

    @pytest.mark.parametrize("model,input_shape,target_shape,loss_fn,"
                            "create_optim_fn,max_range",
                            [   # Base Classifier (Resnet 18)
                                (
                                        ClassificationNetworkBaseTf(1, 10),
                                        # model
                                        (1, 224, 224),  # input shape
                                        9,  # output shape (num_classes - 1)
                                        {"loss_fn": tf.losses.softmax_cross_entropy
                                        },  # loss function
                                        create_optims_default_tf, # optim_fn
                                        4
                                )
                            ])
    @pytest.mark.skipif((virtual_memory().total / 1024.**3) < 20,
                        reason="Less than 20GB of memory")
    def test_tf_model_default(model: ClassificationNetworkBaseTf, input_shape,
                                target_shape, loss_fn, create_optim_fn, max_range):

        start_time = time.time()

        # test backward if optimizer fn is not None
        if create_optim_fn is not None:
            optim = create_optim_fn(tf.train.AdamOptimizer)

        else:
            optim = {}

        model._add_losses(loss_fn)
        model._add_optims(optim)
        model._sess.run(tf.initializers.global_variables())

        closure = model.closure

        # classification label: target_shape specifies max label
        if isinstance(target_shape, int):
            label = np.asarray([np.random.randint(target_shape) for i in range(
                10)])[np.newaxis, :]
        else:
            label = np.random.rand(10, *target_shape) * max_range

        data_dict = {
            "data": np.random.rand(10, *input_shape),
            "label": label
        }

        try:
            closure(model, data_dict)
        except Exception as e:
            assert False, "Test for %s not passed: Error: %s" \
                        % (model.__class__.__name__, e)

        end_time = time.time()

        print("Time needed for %s: %.3f" % (model.__class__.__name__, end_time -
                                            start_time))

        del optim
        del closure
        del model
        gc.collect()


    if __name__ == '__main__':
        # checks if networks are valid (not if they learn something)
        test_tf_model_default()
