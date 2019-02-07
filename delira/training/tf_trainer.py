import os
import logging
import numpy as np
from tqdm.auto import tqdm
import tensorflow as tf
from batchgenerators.dataloading import MultiThreadedAugmenter
from .callbacks import AbstractCallback
from .abstract_trainer import AbstractNetworkTrainer
from .train_utils import create_optims_default_tf as create_optims_default
from .train_utils import initialize_uninitialized
from ..io import tf_load_checkpoint, tf_save_checkpoint
from delira.logging import TrixiHandler
from trixi.logger.tensorboard.tensorboardxlogger import TensorboardXLogger

logger = logging.getLogger(__name__)


class TfNetworkTrainer(AbstractNetworkTrainer):
    """
    Train and Validate a Network

    See Also
    --------
    :class:`AbstractNetwork`

    """

    def __init__(self, network, save_path,
                 losses: dict, optimizer_cls,
                 optimizer_params={}, metrics={}, lr_scheduler_cls=None,
                 lr_scheduler_params={}, gpu_ids=[], save_freq=1,
                 optim_fn=create_optims_default,
                 fold=0, callbacks=[], start_epoch=1,
                 **kwargs):

        """

        Parameters
        ----------
        network : :class:`AbstractTfNetwork`
            the network to train
        save_path : str
            path to save networks to
        losses : dict
            dictionary containing the training losses
        optimizer_cls : subclass of tf.train.Optimizer
            optimizer class implementing the optimization algorithm of choice
        optimizer_params : dict
            keyword arguments passed to optimizer during construction
        metrics : dict
            dictionary containing the validation metrics
        lr_scheduler_cls : Any
            learning rate schedule class: must implement step() method
        lr_scheduler_params : dict
            keyword arguments passed to lr scheduler during construction
        gpu_ids : list
            list containing ids of GPUs to use; if empty: use cpu instead
        save_freq : int
            integer specifying how often to save the current model's state.
            State is saved every state_freq epochs
        optim_fn : function
            creates a dictionary containing all necessary optimizers
        fold : int
            current cross validation fold (0 per default)
        callbacks : list
            initial callbacks to register
        start_epoch : int
            epoch to start training at
        **kwargs :
            additional keyword arguments

        """

        super().__init__(fold, callbacks)

        self.save_path = save_path
        if os.path.isdir(save_path):
            logger.warning(
                "Save Path already exists. Saved Models may be overwritten")
        else:
            os.makedirs(save_path)

        # remove prior Trixihandlers and ensure logging of training results to self.save_path
        # This facilitates visualization of multiple splits/fold inside one tensorboard-instance by means of
        # different tf.Summary.FileWriters()
        root_logger = logging.getLogger()
        for handler in root_logger.handlers:
            handler.close()
        root_logger.handlers = []
        logging.basicConfig(level=logging.INFO,
                            handlers=[TrixiHandler(TensorboardXLogger, 0, self.save_path)])

        self.losses = losses

        self.metrics = metrics

        self.save_freq = save_freq

        # Whether or not to stop the training
        # Used for early stopping
        self.stop_training = False
        self.start_epoch = start_epoch

        self._setup(network, optim_fn, optimizer_cls, optimizer_params,
                    lr_scheduler_cls, lr_scheduler_params, gpu_ids)

        for key, val in kwargs.items():
            setattr(self, key, val)

    def _setup(self, network, optim_fn, optimizer_cls, optimizer_params,
               lr_scheduler_cls, lr_scheduler_params, gpu_ids):
        """
        Defines the Trainers Setup

        Parameters
        ----------
        network : instance of :class: `AbstractTfNetwork`
            the network to train
        optim_fn : function
            creates a dictionary containing all necessary optimizers
        optimizer_cls : subclass of tf.train.Optimizer
            optimizer class implementing the optimization algorithm of choice
        optimizer_params : dict
        lr_scheduler_cls : Any
            learning rate schedule class: must implement step() method
        lr_scheduler_params : dict
            keyword arguments passed to lr scheduler during construction
        gpu_ids : list
            list containing ids of GPUs to use; if empty: use cpu instead
        """

        self.optimizers = optim_fn(optimizer_cls, **optimizer_params)

        # schedulers
        if lr_scheduler_cls is not None:
            for key, optim in self.optimizers.items():
                if not isinstance(lr_scheduler_cls, AbstractCallback):
                    logger.warning("lr_scheduler_cls is not a callback.")
                self.register_callback(lr_scheduler_cls(optim,
                                                        **lr_scheduler_params))

        # asssign closure and prepare batch from network
        self.closure_fn = network.closure

        # TODO: implement multi-GPU and single GPU training with help of
        #  https://www.tensorflow.org/api_docs/python/tf/keras/utils/multi_gpu_model
        #  note: might be bugged in combination with sess.run https://github.com/tensorflow/tensorflow/issues/21788
        #  and https://www.tensorflow.org/api_docs/python/tf/keras/models/clone_model

        """
        if gpu_ids and tf.test.is_gpu_available():
            assert len(gpu_ids) <= len(get_available_gpus()), "more GPUs specified than available"
            self.use_gpu = True
            if len(gpu_ids) > 1:
                logger.warning(
                    "multi-GPU training not yet tested!")

                network.model = tf.keras.utils.multi_gpu_model(
                                        network.model,
                                        len(gpu_ids),
                                        cpu_merge=True,
                                        cpu_relocation=False)
            else:
                network.models = tf.keras.models.clone_model(network.model)
        else:
            self.use_gpu = False
        """
        self.use_gpu = True
        self.module = network

        # TODO: Beautify
        self.module._add_losses(self.losses)
        self.module._add_optims(self.optimizers)
        # check for unitialized variables
        initialize_uninitialized(self.module._sess)

    def train(self, num_epochs, datamgr_train, datamgr_valid=None,
              val_score_key=None, val_score_mode='highest'):
        """
        train network

        Parameters
        ----------
        num_epochs : int
            number of epochs to train
        datamgr_train : BaseDataManager
            Data Manager to create Batch Generator for training
        datamgr_valid : BaseDataManager
            Data Manager to create Batch Generator for validation
        val_score_key : str
            Key of validation metric; must be key in self.metrics
        val_score_mode : str
            String to specify whether a higher or lower validation score is
            optimal; must be in ['highest', 'lowest']

        Returns
        -------
        :class:`AbstractTfNetwork`
            Best model (if `val_score_key` is not a valid key the model of the
            last epoch will be returned)

        """

        self._at_training_begin()

        if val_score_mode == 'highest':
            best_val_score = 0
        elif val_score_mode == 'lowest':
            best_val_score = float('inf')
        else:
            best_val_score = None

        curr_val_score = best_val_score

        self.save_state(os.path.join(self.save_path, "checkpoint_epoch_0"))
        metrics_val = {}

        for epoch in range(self.start_epoch, num_epochs+1):

            self._at_epoch_begin(metrics_val, val_score_key, epoch,
                                 num_epochs)

            batch_gen_train = datamgr_train.get_batchgen(seed=epoch)

            self._train_single_epoch(batch_gen_train, epoch)

            if datamgr_valid:
                # validate with batchsize 1 and 1 augmentation processs to
                # avoid dropping of last elements
                orig_num_aug_processes = datamgr_valid.n_process_augmentation
                orig_batch_size = datamgr_valid.batch_size

                datamgr_valid.batch_size = 1
                datamgr_valid.n_process_augmentation = 1

                labels_val, pred_val, metrics_val = self.predict(
                    datamgr_valid.get_batchgen(), batch_size=orig_batch_size)

                # reset old values
                datamgr_valid.batch_size = orig_batch_size
                datamgr_valid.n_process_augmentation = orig_num_aug_processes

                # ToDO: Move decision, if current model is best to callback
                if val_score_key in metrics_val.keys():
                    curr_val_score = metrics_val[val_score_key]
                    is_best = self._is_better_val_scores(best_val_score,
                                                         curr_val_score,
                                                         val_score_mode)

                else:
                    logger.warning(
                        "Validation score key not in metric dict. "
                        "Logging metrics but can't decide which model is best")

                    is_best = False

                if is_best:
                    best_val_score = curr_val_score
                    tqdm.write(
                        'Best val score = %2.3f' % best_val_score.item())
                else:
                    is_best = False
            else:
                is_best = False
                labels_val, pred_val, metrics_val = {}, {}, {}

            self._at_epoch_end(metrics_val, val_score_key, epoch, is_best)

            # stop training (might be caused by early stopping)
            if self.stop_training:
                break

        return self._at_training_end()

    def _at_training_begin(self, *args, **kwargs):
        """
        Defines behaviour at beginning of training

        Parameters
        ----------
        *args :
            positional arguments
        **kwargs :
            keyword arguments

        """
        pass

    def _at_training_end(self):
        """
        Defines Behaviour at end of training: Loads best model if available

        Returns
        -------
        :class:`AbstractTfNetwork`
            best network

        """
        if os.path.isfile(os.path.join(self.save_path, 'checkpoint_best.meta')):

            # load best model and return it. Since the state is hidden in the graph, we don't actually need to use
            # self.update_state.
            self.update_state(os.path.join(self.save_path,
                                           'checkpoint_best')
                              )

        return self.module

    def _at_epoch_begin(self, metrics_val, val_score_key, epoch, num_epochs,
                        **kwargs):
        """
        Defines behaviour at beginning of each epoch: Executes all callbacks's
        `at_epoch_begin` method

        Parameters
        ----------
        metrics_val : dict
            validation metrics
        val_score_key : str
            validation score key
        epoch : int
            current epoch
        num_epochs : int
            total number of epochs
        **kwargs :
            keyword arguments

        """

        # execute all callbacks
        for cb in self._callbacks:
            self._update_state(cb.at_epoch_begin(self, val_metrics=metrics_val,
                                                 val_score_key=val_score_key,
                                                 curr_epoch=epoch))

    def _at_epoch_end(self, metrics_val, val_score_key, epoch, is_best,
                      **kwargs):
        """
        Defines behaviour at beginning of each epoch: Executes all callbacks's
        `at_epoch_end` method and saves current state if necessary

        Parameters
        ----------
        metrics_val : dict
            validation metrics
        val_score_key : str
            validation score key
        epoch : int
            current epoch
        num_epochs : int
            total number of epochs
        **kwargs :
            keyword arguments

        """

        for cb in self._callbacks:
            self._update_state(cb.at_epoch_end(self, val_metrics=metrics_val,
                                               val_score_key=val_score_key,
                                               curr_epoch=epoch))

        if epoch % self.save_freq == 0:
            self.save_state(os.path.join(self.save_path,
                                         "checkpoint_epoch_%d" % epoch))

        if is_best:
            self.save_state(os.path.join(self.save_path,
                                         "checkpoint_best"))

    def _train_single_epoch(self, batchgen: MultiThreadedAugmenter, epoch):
        """
        Trains the network a single epoch

        Parameters
        ----------
        batchgen : MultiThreadedAugmenter
            Generator yielding the training batches
        epoch : int
            current epoch

        """
        self.module.training = True

        n_batches = batchgen.generator.num_batches * batchgen.num_processes
        pbar = tqdm(enumerate(batchgen), unit=' batch', total=n_batches,
                    desc='Epoch %d' % epoch)

        for batch_nr, batch in pbar:

            data_dict = batch

            _, _, _ = self.closure_fn(self.module, data_dict,
                                      optimizers=self.optimizers,
                                      losses=self.losses,
                                      metrics=self.metrics,
                                      fold=self.fold,
                                      batch_nr=batch_nr)

        batchgen._finish()

    def predict(self, batchgen, batch_size=None):
        """
        Returns predictions from network for batches from batchgen

        Parameters
        ----------
        batchgen : MultiThreadedAugmenter
            Generator yielding the batches to predict

        batch_size : None or int
            if int: collect batches until batch_size is reached and
            forward them together

        Returns
        -------
        np.ndarray
            predictions from batches
        list of np.ndarray
            labels from batches
        dict
            dictionary containing the mean validation metrics and
            the mean loss values

        """
        self.module.training = False

        outputs_all, labels_all = [], []
        metric_mean_vals = {}
        loss_mean_vals = {}

        n_batches = batchgen.generator.num_batches * batchgen.num_processes

        pbar = tqdm(enumerate(batchgen), unit=' sample',
                    total=n_batches, desc='Test')

        batch_list = []

        orig_batch_size = batch_size

        for i, batch in pbar:

            if not batch_list and (n_batches - i) < batch_size:

                batch_size = n_batches - i
                logger.debug("Set Batchsize down to %d to avoid cutting "
                             "of the last batches" % batch_size)

            # queue inputs and labels
            batch_list.append(batch)

            # if queue is full process queue:
            if batch_size is None or len(batch_list) >= batch_size:

                batch_dict = {}
                for batch in batch_list:
                    for key, val in batch.items():
                        if key in batch_dict.keys():
                            batch_dict[key].append(val)
                        else:
                            batch_dict[key] = [val]

                for key, val_list in batch_dict.items():
                    batch_dict[key] = np.concatenate(val_list)

                met_vals, loss_vals, preds = self.closure_fn(
                    self.module, batch_dict,
                    optimizers={},
                    losses=self.losses,
                    metrics=self.metrics,
                    fold=self.fold)

                for key, val in met_vals.items():

                    if key in metric_mean_vals.keys():
                        metric_mean_vals[key] += val
                    else:
                        metric_mean_vals[key] = val

                for key, val in loss_vals.items():

                    if key in loss_mean_vals.keys():
                        loss_mean_vals[key] += val
                    else:
                        loss_mean_vals[key] = val

                outputs_all.append(tmp for tmp in preds)

                label_dict = {}

                for key, val in batch_dict.items():
                    if "data" not in key and "img" not in key:
                        label_dict[key] = val

                labels_all.append([label_dict[key]
                                   for key in sorted(label_dict.keys())])

                batch_list = []

        batchgen._finish()

        # transpose labels and outputs to have a list of lists of
        # labels of same type
        labels_all = zip(*labels_all)
        outputs_all = zip(*outputs_all)

        labels_all = [np.vstack(_labels) for _labels in labels_all]
        outputs_all = [np.vstack(_outputs) for _outputs in outputs_all]

        # metric_mean_vals contains sums of metrics so far.
        # Dividing by number of batches to get mean values

        # if virtual batchsize is given: calculate actual number of batches
        if batch_size is not None:
            div = np.ceil(n_batches / orig_batch_size)
        else:
            div = n_batches

        val_dict = {}
        for key, val in metric_mean_vals.items():
            val_dict[key] = val / div

        for key, val in loss_mean_vals.items():
            val_dict[key] = val / div

        return outputs_all, labels_all, val_dict

    def save_state(self, file_name):
        """
        saves the current state via :func:`delira.io.tf.save_checkpoint`

        Parameters
        ----------
        file_name : str
            filename to save the state to
        """
        tf_save_checkpoint(file_name, self.module)


    def load_state(self, file_name):
        """
        Loads the new state from file via :func:`delira.io.tf.load_checkpoint`

        Parameters
        ----------
        file_name : str
            the file to load the state from
        Returns
        -------

        """
        return tf_load_checkpoint(file_name, self.module)
