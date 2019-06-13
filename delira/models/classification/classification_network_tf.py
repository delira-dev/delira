import logging
import typing

import tensorflow as tf

from delira.models.abstract_network import AbstractTfNetwork
from delira.models.classification.ResNet18 import ResNet18

from delira.utils.decorators import make_deprecated

logger = logging.getLogger(__name__)


class ClassificationNetworkBaseTf(AbstractTfNetwork):
    """
    Implements basic classification with ResNet18

    See Also
    --------
    :class:`AbstractTfNetwork`

    """

    @make_deprecated("own repository to be announced")
    def __init__(self, in_channels: int, n_outputs: int, **kwargs):
        """

        Constructs graph containing model definition and forward pass behavior

        Parameters
        ----------
        in_channels : int
            number of input_channels
        n_outputs : int
            number of outputs (usually same as number of classes)
        """
        tf.keras.backend.set_image_data_format('channels_first')
        # register params by passing them as kwargs to parent class __init__
        super().__init__(in_channels=in_channels,
                         n_outputs=n_outputs,
                         **kwargs)

        # build on CPU for tf.keras.utils.multi_gpu_model() in tf_trainer.py
        # with tf.device('/cpu:0'):
        self.model = self._build_model(n_outputs, **kwargs)

        images = tf.placeholder(shape=[None, in_channels, None, None],
                                dtype=tf.float32)
        labels = tf.placeholder(shape=[None, n_outputs], dtype=tf.float32)

        preds_train = self.model(images, training=True)
        preds_eval = self.model(images, training=False)

        self.inputs["images"] = images
        self.inputs["labels"] = labels
        self.outputs_train["pred"] = preds_train
        self.outputs_eval["pred"] = preds_eval

        for key, value in kwargs.items():
            setattr(self, key, value)

    def _add_losses(self, losses: dict):
        """
        Adds losses to model that are to be used by optimizers or during
        evaluation

        Parameters
        ----------
        losses : dict
            dictionary containing all losses. Individual losses are averaged
        """
        if self._losses is not None and len(losses) != 0:
            logging.warning('Change of losses is not yet supported')
            raise NotImplementedError()
        elif self._losses is not None and len(losses) == 0:
            pass
        else:
            self._losses = {}
            for name, _loss in losses.items():
                self._losses[name] = _loss(self.inputs['labels'],
                                           self.outputs_train['pred'])

            total_loss = tf.reduce_mean(list(self._losses.values()), axis=0)

            self._losses['total'] = total_loss
            self.outputs_train['losses'] = self._losses
            self.outputs_eval['losses'] = self._losses

    def _add_optims(self, optims: dict):
        """
        Adds optims to model that are to be used by optimizers or during
        training

        Parameters
        ----------
        optim: dict
            dictionary containing all optimizers, optimizers should be of
            Type[tf.train.Optimizer]
        """
        if self._optims is not None and len(optims) != 0:
            logging.warning('Change of optims is not yet supported')
            pass
            # raise NotImplementedError()
        elif self._optims is not None and len(optims) == 0:
            pass
        else:
            self._optims = optims['default']
            grads = self._optims.compute_gradients(self._losses['total'])
            step = self._optims.apply_gradients(grads)
            self.outputs_train['default_optim'] = step

    @staticmethod
    def _build_model(n_outputs: int, **kwargs):
        """
        builds actual model (resnet 18)

        Parameters
        ----------
        n_outputs : int
            number of outputs (usually same as number of classes)
        **kwargs :
            additional keyword arguments
        Returns
        -------
        tf.keras.Model
            created model
        """
        model = ResNet18(num_classes=n_outputs)

        return model

    @staticmethod
    def closure(model: typing.Type[AbstractTfNetwork], data_dict: dict,
                metrics=None, fold=0, **kwargs):
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
            Loss values (with same keys as those initially passed to
            model.init).
            Additionally, a total_loss key is added
        dict
            outputs of `model.run`

        """

        if metrics is None:
            metrics = {}
        loss_vals = {}
        metric_vals = {}

        inputs = data_dict.pop('data')

        outputs = model.run(images=inputs, labels=data_dict['label'])
        preds = outputs['pred']
        losses = outputs['losses']

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

        return metric_vals, loss_vals, outputs
