import logging
import torch
import numpy as np
from visdom import Visdom
from delira.utils import now
from delira.utils.decorators import make_deprecated


from ..trixi_handler import TrixiHandler


@make_deprecated(TrixiHandler)
class VisdomImageHandler(logging.Handler):
    """
    Logging Handler to show images and metric plots with visdom

    .. deprecated:: 0.1
        :class:`VisdomImageHandler` will be removed in next release and is
        deprecated in favor of ``trixi.logging`` Modules

    .. warning::
        :class:`VisdomImageHandler` will be removed in next release

    See Also
    --------
    `Visdom`
    :class:`TrixiHandler`

    """
    def __init__(self, port, prefix, log_freq_train, log_freq_val=1e10,
                 level=logging.NOTSET, log_freq_img=1, **kwargs):
        """

        Parameters
        ----------
        port: int
            port of visdom-server
        prefix : str
            prefix of environment names
        log_freq_train : int
            Defines logging frequency for scores in train mode
        log_freq_val : int
            Defines logging frequency for scores in validation mode
        level : int (default: logging.NOTSET)
            logging level
        **kwargs:
            additional keyword arguments which are directly passed to visdom

        """
        super().__init__(level=level)

        self.viz = Visdom(port=port, env=prefix, **kwargs)
        self.env_prefix = prefix
        self.log_freq_train = log_freq_train
        self.log_freq_val = log_freq_val
        self.curr_batch_train = 1
        self.curr_batch_val = 1
        self.curr_epoch_train = 1
        self.curr_epoch_val = 1
        self.metrics = {}
        self.val_metrics = {}
        self.plot_windows = {}
        self.image_windows = {}
        self.heatmap_windows = {}
        self.text_windows = {}
        self.bar_windows = {}
        self.curr_env_name = prefix
        self.curr_fold = None
        self.img_count = 0
        self.log_freq_img = log_freq_img

    def emit(self, record):
        """
        shows images and metric plots in visdom

        Parameters
        ----------
        record : LogRecord
            entities to log

        Returns
        -------
        None
            * if no connection to `visdom` could be found
            * if `record.msg` is not a dict

        """
        # messages that cant be send fill (GPU-)RAM so return if no connection
        if not self.viz.check_connection():
            return

        if not isinstance(record.msg, dict):
            return

        scores = record.msg.get("scores", {})
        images = record.msg.get("images", {})
        heatmaps = record.msg.get("heatmaps", {})
        scalars = record.msg.get("scalars", {})
        bars = record.msg.get("bars", {})
        fold = record.msg.get("fold", "")
        text = record.msg.get("text", {})
        plots = record.msg.get("plots", {})

        if fold != self.curr_fold:
            self.curr_batch_train = 1
            self.curr_batch_val = 1
            self.curr_epoch_train = 1
            self.curr_epoch_val = 1
            self.metrics = {}
            self.val_metrics = {}
            self.plot_windows = {}
            self.image_windows = {}
            self.heatmap_windows = {}
            self.text_windows = {}
            self.bar_windows = {}

            if not fold and isinstance(fold, str):
                fold_name = self.env_prefix

            else:
                fold_name = self.env_prefix + "_fold_%02d_%s" % (fold, now())

        else:
            fold_name = self.curr_env_name

        self.curr_fold = fold
        self.curr_env_name = fold_name

        # Log losses and metrics
        for i, metric_name in enumerate(scores.keys()):

            # handle validation scores
            if metric_name.startswith("val_"):
                metric_name = metric_name.split("_", maxsplit=1)[-1]
                if metric_name not in self.val_metrics:
                    self.val_metrics[metric_name] = self._to_scalar(
                        scores["val_" + metric_name])
                else:
                    self.val_metrics[metric_name] += self._to_scalar(
                        scores["val_" + metric_name])

            # handle train scores
            else:
                if metric_name not in self.metrics:
                    self.metrics[metric_name] = self._to_scalar(
                        scores[metric_name])
                else:
                    self.metrics[metric_name] += self._to_scalar(
                        scores[metric_name]
                    )

        # Draw images
        self.img_count += 1
        if (self.img_count % self.log_freq_img) == 0:
            for image_name, tensor in images.items():
                if image_name not in self.image_windows:
                    self.image_windows[image_name] = self.viz.image(
                        self._to_image(tensor),
                        opts={'title': image_name},
                        env=fold_name)
                else:
                    self.viz.image(self._to_image(tensor.data),
                                   win=self.image_windows[image_name],
                                   opts={'title': image_name},
                                   env=fold_name)
            self.img_count = 0

        # draw heatmaps
        for heatmap_name, tensor in heatmaps.items():
            heatmap = tensor[0].cpu().numpy()
            if heatmap_name not in self.heatmap_windows:
                self.heatmap_windows[heatmap_name] = self.viz.heatmap(
                    heatmap, opts=dict(title=heatmap_name, colormap='hot'),
                    env=fold_name)
            else:
                self.viz.heatmap(heatmap,
                                 win=self.heatmap_windows[heatmap_name],
                                 opts=dict(title=heatmap_name, colormap='hot'),
                                 env=fold_name)

        # visualize scalars
        for scalar_name, scalar_val in scalars.items():
            text_str = "<font face = 'Arial' size = '4'>%s</font>" % \
                       str(self._to_scalar(scalar_val))
            if scalar_name not in self.text_windows:
                self.text_windows[scalar_name] = self.viz.text(text_str,
                                                               env=fold_name)
            else:
                self.viz.text(text_str, win=self.text_windows[scalar_name],
                              env=fold_name)

        # draw bars
        for bar_name, bar_vals in bars.items():
            if bar_name not in self.bar_windows:
                self.bar_windows[bar_name] = self.viz.bar(
                    bar_vals, opts={"title": bar_name},
                    env=fold_name)

            else:
                self.viz.bar(bar_vals, win=self.bar_windows[bar_name],
                             opts={"title": bar_name},
                             env=fold_name)

        # visualize text
        for text_name, val_str in text.items():
            text_str = "<font face = 'Arial' size = '4'>%s</font>" % val_str

            if text_name not in self.text_windows:
                self.text_windows[text_name] = self.viz.text(text_str,
                                                             env=fold_name)
            else:
                self.viz.text(text_str, win=self.text_windows[text_name],
                              env=fold_name)

        # visualize plots
        for plot_name, plot_vals in plots.items():

            if isinstance(plot_vals, dict):
                x_vals = plot_vals["x"]
                y_vals = plot_vals["y"]
                xlabel = plot_vals.get("xlabel", "")
                ylabel = plot_vals.get("ylabel", "")
            else:
                x_vals = np.array(plot_vals[0])
                y_vals = np.array(plot_vals[1])
                xlabel = ""
                ylabel = ""

            if plot_name not in self.plot_windows:
                self.plot_windows[plot_name] = self.viz.line(
                    X=x_vals,
                    Y=y_vals,
                    opts={'xlabel': xlabel,
                          'ylabel': ylabel,
                          'title': plot_name},
                    env=fold_name)

            else:
                self.viz.line(X=x_vals,
                              Y=y_vals,
                              win=self.plot_windows[plot_name],
                              opts={'xlabel': xlabel,
                                    'ylabel': ylabel,
                                    'title': plot_name},
                              env=fold_name)

        # End of epoch
        # decide which dict to log
        # only one epoch type at same type possible

        # train epoch ended
        if (self.curr_batch_train % self.log_freq_train) == 0:
            score_dict = self.metrics
            curr_batch = self.curr_batch_train
            curr_epoch = self.curr_epoch_train
            name = "train"
            self.curr_epoch_train += 1
            self.curr_batch_train = 1
            self.metrics = {}

        # validation epoch ended
        elif (self.curr_batch_val % self.log_freq_val) == 0:
            score_dict = self.val_metrics
            curr_batch = self.curr_batch_val
            curr_epoch = self.curr_epoch_val
            name = "val"
            self.curr_epoch_val += 1
            self.curr_batch_val = 1
            self.val_metrics = {}

        # no epoch ended
        else:
            score_dict = {}
            curr_epoch = 1
            curr_batch = 1

        if score_dict:
            # Plot losses
            for metric_name, metric in score_dict.items():
                if metric_name not in self.plot_windows:
                    self.plot_windows[metric_name] = self.viz.line(
                        X=np.array([curr_epoch]),
                        Y=np.array([metric / curr_batch]),
                        opts={'xlabel': 'iterations',
                                'ylabel': metric_name,
                                'title': metric_name}, name=name,
                        env=fold_name)

                else:
                    self.viz.line(X=np.array([curr_epoch]),
                                  Y=np.array([metric / curr_batch]),
                                  win=self.plot_windows[metric_name],
                                  update='append', name=name,
                                  env=fold_name)

        else:
            is_val = False
            is_train = False
            for key in scores.keys():
                if key.startswith("val_"):
                    is_val = True
                else:
                    is_train = True

            if is_val:
                self.curr_batch_val +=1

            if is_train:
                self.curr_batch_train += 1

    @staticmethod
    def _to_scalar(val):
        """
        convert scalar wrapped in tensor or numpy array to float
        Parameters
        ----------
        val: torch.Tensor or numpy array
            value to be converted

        Returns
        -------
        float
            converted value

        """
        if isinstance(val, np.ndarray):
            return np.asscalar(val)
        elif isinstance(val, torch.Tensor):
            return val.item()
        else:
            return float(val)

    @staticmethod
    def _to_image(tensor: torch.Tensor):
        """
        convert image to numpy array
        Parameters

        ----------
        tensor: entity which is convertible to numpy array
            image tensor

        Returns
        -------
        np.ndarray
            converted image

        """
        img = tensor[0].cpu().numpy()

        if img.shape[0] == 1:
            img = np.tile(img, (3, 1, 1))

        img -= img.min()
        if img.max():
            img *= 255/img.max()

        return img.astype(np.uint8)
