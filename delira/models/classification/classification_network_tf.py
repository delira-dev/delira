import logging
import numpy as np
import tensorflow as tf
from tqdm import tqdm
from sklearn.metrics import accuracy_score as acc
from typing import Type
from torchvision import models as t_models
from delira.models.abstract_network import AbstractTfNetwork
from ResNet18 import ResNet18

tf.keras.backend.set_image_data_format('channels_first')


file_logger = logging.getLogger(__name__)


class ClassificationNetworkBaseTf(AbstractTfNetwork):
    """
    Implements basic classification with ResNet18

    See Also
    --------
    :class:`AbstractTfNetwork`

    """

    def __init__(self, in_channels: int, n_outputs: int, losses: dict,
                 optim: Type[tf.train.Optimizer], **kwargs):
        """

        Constructs graph containing model definition, forward pass,
        loss functions and optimizers.

        Parameters
        ----------
        in_channels : int
            number of input_channels
        n_outputs : int
            number of outputs (usually same as number of classes)
        losses : dict
            dictionary containing all losses. Individual losses are averaged
        optim: tf.train.Optimizer
            single optimizer, called on total_loss
        """
        # register params by passing them as kwargs to parent class __init__
        super().__init__(in_channels=in_channels,
                         n_outputs=n_outputs,
                         **kwargs)

        #with tf.device('CPU:0'):
        #    self.module = self._build_model(n_outputs, **kwargs)
        #self.module = tf.keras.utils.multi_gpu_model(self.module, 1)

        self.module = self._build_model(n_outputs, **kwargs)

        images = tf.placeholder(shape=[None, in_channels, None, None],
                                dtype=tf.float32)
        labels = tf.placeholder(shape=[None, n_outputs], dtype=tf.float32)

        preds_train = self.module(images, training=True)
        preds_eval = self.module(images, training=False)

        loss = []
        for name, _loss in losses.items():
            losses[name] = _loss(labels, preds_train)
            loss.append(losses[name])

        total_loss = tf.reduce_mean(loss, axis=0)

        losses['total'] = total_loss

        _optim = optim()
        grads = _optim.compute_gradients(total_loss)
        step = _optim.apply_gradients(grads)

        self.inputs = [images, labels]
        self.outputs_train = [preds_train, losses, step]
        self.outputs_eval = [preds_eval, losses]
        self.sess.run(tf.initializers.global_variables())

        for key, value in kwargs.items():
            setattr(self, key, value)

    @staticmethod
    def _build_model(n_outputs: int):
        """
        builds actual model (resnet 18)

        Parameters
        ----------
        n_outputs : int
            number of outputs (usually same as number of classes)

        Returns
        -------
        tf.keras.Model
            created model
        """
        model = ResNet18(num_classes=n_outputs)

        return model

    @staticmethod
    def closure(model: Type[AbstractTfNetwork], data_dict: dict,
                metrics={}, fold=0, **kwargs):
        """
                closure method to do a single prediction.
                This is followed by backpropagation or not based state of
                on model.train

                Parameters
                ----------
                model: AbstractTfNetwork
                    AbstractTfNetwork or its child-clases
                data_dict : dict
                    dictionary containing the data
                metrics : dict
                    dict holding the metrics to calculate
                fold : int
                    Current Fold in Crossvalidation (default: 0)
                **kwargs:
                    additional keyword arguments

                Returns
                -------
                dict
                    Metric values (with same keys as input dict metrics)
                dict
                    Loss values (with same keys as those initially passed to model.init).
                    Additionally, a total_loss key is added
                list
                    Arbitrary number of predictions as np.array

                """

        loss_vals = {}
        metric_vals = {}

        inputs = data.pop('data')

        preds, losses, *_ = model.run(inputs, data_dict['label'])

        for key, loss_val in losses.items():
            loss_vals[key] = loss_val

        for key, metric_fn in metrics.items():
            metric_vals[key] = metric_fn(
                preds, *data_dict.values())

        if not model.training:
            # add prefix "val" in validation mode
            eval_loss_vals, eval_metrics_vals = {}, {}
            for key in loss_vals.keys():
                eval_loss_vals["val_" + str(key)] = loss_vals[key]

            for key in metric_vals:
                eval_metrics_vals["val_" + str(key)] = metric_vals[key]

            loss_vals = eval_loss_vals
            metric_vals = eval_metrics_vals

        for key, val in {**metric_vals, **loss_vals}.items():
            logging.info({"value": {"value"[0]: val.item(), "name": key,
                                    "env_appendix": "_%02d" % fold
                                    }})

        logging.info({'image_grid': {"images": inputs, "name": "input_images",
                                     "env_appendix": "_%02d" % fold}})

        return metric_vals, loss_vals, [preds]

if __name__ == '__main__':
    l = {}
    l['softCE'] = tf.losses.softmax_cross_entropy
    l['sigCE'] = tf.losses.sigmoid_cross_entropy
    asd = ClassificationNetworkBaseTf(3, 7, losses=l, optim=tf.train.AdamOptimizer)

    def accuracy_score(y_true, y_pred):
        y_true = np.argmax(y_true, axis=0)
        y_pred = np.argmax(y_pred, axis=0)
        return acc(y_true, y_pred)
    for i, _ in enumerate(tqdm(range(1000))):
        data = {}
        data['data'] = np.random.rand(10, 3, 224, 224)
        data['label'] = np.random.random_integers(0, 1, size=(10, 7))

        metrics = {}
        metrics['Accuracy'] = accuracy_score

        asd.train()
        asd.closure(asd, data, metrics)

        data = {}
        data['data'] = np.random.rand(100, 3, 224, 224)
        data['label'] = np.random.random_integers(0, 1, size=(100, 7))

        asd.eval()
        asd.closure(asd, data, metrics)
