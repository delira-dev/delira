import logging
import torch
from skimage import io as sio
import numpy as np
import os

from delira.utils.decorators import make_deprecated

from ..trixi_handler import TrixiHandler


@make_deprecated(TrixiHandler)
class ImgSaveHandler(logging.Handler):
    """
    Logging Handler which saves images to dir

    .. deprecated:: 0.1
        :class:`ImgSaveHandler` will be removed in next release and is
        deprecated in favor of ``trixi.logging`` Modules

    .. warning::
        :class:`ImgSaveHandler` will be removed in next release

    See Also
    --------
    :class:`TrixiHandler`

    """

    def __init__(self, save_dir_train, save_dir_val=None, save_freq_train=1,
                 save_freq_val=1, level=logging.NOTSET):
        """
        Parameters
        ----------
        save_dir_train : str
            path to which the training images should be saved (must not yet be
            existent)
        save_dir_val : str (default:None)
            path to which the training images should be saved (must not yet be
            existent)
        save_freq_train : int (default: 1)
            frequency with which images are saved during training
        save_freq_val : int (default: 1)
            frequency with which images are saved during validation
        level: int (default: logging.NOTSET)
            logging level

        Raises
        ------
        DeprecationWarning
            First Time a class instance is created

        """
        super().__init__(level)
        self._save_dir_train = save_dir_train
        self._save_dir_val = save_dir_val
        self._curr_index_train = 0
        self._curr_index_val = 0

        def _set_save_freq(name, save_freq):
            assert save_freq > 0
            assert isinstance(save_freq, int)
            setattr(self, name, save_freq)

        _set_save_freq("save_freq_train", save_freq_train)
        _set_save_freq("save_freq_val", save_freq_val)
        self.curr_batch_train = 0
        self.curr_batch_val = 0
        os.makedirs(save_dir_train, exist_ok=True)
        if save_dir_val:
            os.makedirs(save_dir_val, exist_ok=True)

    def emit(self, record):
        """
        Logging record message

        Parameters
        ----------
        record : LogRecord
            values to log

        Returns
        -------
        None
            if `record.msg` is not a dict

        """
        save_imgs = False
        if not isinstance(record.msg, dict):
            try:
                img = self._to_image(record.msg)
                if self.curr_batch_train % self.save_freq_train == 0:
                    self._save_image_batch(img, "image_%05d" %
                                           self._curr_index_train)
                    self._curr_index_train += 1
            except Exception as e:
                pass

            return

        images = record.msg.get('images', {})
        scores = record.msg.get('scores', {})
        image_dict = {}

        is_train = not any([name.startswith("val_") for name in scores.keys()])

        if isinstance(images, list):
            if is_train:
                self.curr_batch_train += 1
                if self.curr_batch_train % self.save_freq_train == 0:
                    save_imgs = True
            else:
                self.curr_batch_val += 1
                if self.curr_batch_val % self.save_freq_val == 0:
                    save_imgs = True
            if save_imgs:
                for img in images:
                    if is_train:
                        curr_index = self._curr_index_train
                    else:
                        curr_index = self._curr_index_val
                    image_dict['image_%05d' % curr_index] = img

        elif isinstance(images, dict):
            if (images):
                if is_train:
                    self.curr_batch_train += 1
                    if self.curr_batch_train % self.save_freq_train == 0:
                        save_imgs = True
                else:
                    self.curr_batch_val += 1
                    if self.curr_batch_val % self.save_freq_val == 0:
                        save_imgs = True
                if save_imgs:
                    for key, img in images.items():
                        if is_train:
                            curr_index = self._curr_index_train
                        else:
                            curr_index = self._curr_index_val

                        new_key = key.replace("val_", "")
                        image_dict[new_key + '_%05d' % curr_index] = img

        if save_imgs:
            if is_train:
                self._curr_index_train += 1
            else:
                self._curr_index_val += 1
            for prefix, batch in image_dict.items():
                self._save_image_batch(batch, prefix, is_train)

    def _save_image_batch(self, batch, prefix, is_train=True):
        """
        Saving image batch to save_dir

        Parameters
        ----------
        batch: iterable
            batch of images
        prefix: str
            file-prefix

        """
        save_dir = self._save_dir_train if is_train else self._save_dir_val
        if isinstance(batch, torch.Tensor):
            batch_elements = [tmp for tmp in batch.split(1)]
        else:
            batch_elements = list(batch)

        for idx, img in enumerate(batch_elements):
            sio.imsave(os.path.join(save_dir, prefix + "_%d.png" % idx),
                       self._to_image(img))

    @staticmethod
    def _to_image(tensor):
        """
        convert image to numpy array

        Parameters
        ----------
        tensor: entity which is convertible to numpy array
            image tensor
        Returns
        -------
        np.ndarray
            converted tensor

        """
        if isinstance(tensor, torch.Tensor):
            img = tensor[0].cpu().numpy()
        else:
            img = np.asarray(tensor)

        img = img.astype(np.float32)

        if img.shape[0] == 1:
            img = np.tile(img, (3, 1, 1))

        img -= img.min()
        if img.max():
            img /= img.max()

        return img.transpose(1, 2, 0)
