from delira.logging import Logger, TensorboardBackend, make_logger

import unittest

try:
    import torch
except ImportError:
    torch = None

try:
    import tensorflow as tf
except ImportError:
    tf = None

try:
    import onnx
except ImportError:
    onnx = None

import numpy as np
import os
import gc


class TestTensorboardLogging(unittest.TestCase):

    def setUp(self) -> None:

        self._npy_imgs = np.random.rand(2, 3, 24, 24)
        self._boxes_npy = np.array([[5, 5, 10, 10], [4, 8, 5, 16]])
        self._scalars = [{"1": 4, "2": 14, "3": 24},
                         {"1": 5, "2": 15, "3": 25},
                         {"1": 6, "2": 16, "3": 26}]

        self._hist_vals = np.random.randint(0, 10, size=(100,))
        from scipy.signal import chirp
        self._audio_sample_npy = chirp(np.linspace(0, 100), 500, 2, 100)

        self._text_string = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrs" \
                            "tuvwxyz0123456789"

        if tf is not None:
            input_shape = [1, 28, 28]

            l = tf.keras.layers
            max_pool = l.MaxPooling2D(
                (2, 2), (2, 2), padding='same', data_format="channels_first")
            self._model_tf = tf.keras.Sequential(
                [
                    l.Reshape(
                        target_shape=input_shape,
                        input_shape=(28 * 28,)),
                    l.Conv2D(
                        32,
                        5,
                        padding='same',
                        data_format="channels_first",
                        activation=tf.nn.relu),
                    max_pool,
                    l.Conv2D(
                        64,
                        5,
                        padding='same',
                        data_format="channels_first",
                        activation=tf.nn.relu),
                ])
        else:
            self._model_tf = None

        if torch is not None:
            self._model_torch = torch.nn.Sequential(
                torch.nn.Conv2d(3, 8, 3, padding=1),
                torch.nn.ReLU(),
                torch.nn.Conv2d(8, 1, 3, padding=1),
                torch.nn.LeakyReLU(),
                torch.nn.Conv2d(1, 23, 3),
            )

        else:
            self._model_torch = None

        self._embedding_npy = np.random.rand(500, 3)

        self._labels_npy = np.random.randint(0, 10, 100)
        self._predictions_npy = np.random.randint(0, 10, 100)

        self._logger = self._setup_logger()

    def _setup_logger(self):
        return make_logger(TensorboardBackend(
            {"log_dir": os.path.join(".", "runs", self._testMethodName)}
        ))

    @staticmethod
    def _destroy_logger(logger: Logger):
        logger.close()
        del logger
        gc.collect()

    def test_image_npy(self):
        self._logger.log({"image": {"tag": "image_npy",
                                    "img_tensor": self._npy_imgs[0]}})

    @unittest.skipIf(torch is None, "Torch Backend not installed")
    def test_image_torch(self):
        self._logger.log({"image": {"tag": "image_torch",
                                    "img_tensor":
                                        torch.from_numpy(self._npy_imgs[0])}})

    def test_img_npy(self):
        self._logger.log({"img": {"tag": "img_npy",
                                  "img_tensor": self._npy_imgs[0]}})

    @unittest.skipIf(torch is None, "Torch Backend not installed")
    def test_img_torch(self):
        self._logger.log({"img": {"tag": "img_torch",
                                  "img_tensor":
                                      torch.from_numpy(self._npy_imgs[0])}})

    def test_picture_npy(self):
        self._logger.log({"picture": {"tag": "picture_npy",
                                      "img_tensor": self._npy_imgs[0]}})

    @unittest.skipIf(torch is None, "Torch Backend not installed")
    def test_picture_torch(self):
        self._logger.log({"picture": {"tag": "picture_torch",
                                      "img_tensor": torch.from_numpy(self._npy_imgs[0])}})

    def test_images_npy(self):
        self._logger.log({"images": {"tag": "images_npy",
                                     "img_tensor": self._npy_imgs}})

    @unittest.skipIf(torch is None, "Torch Backend not installed")
    def test_images_torch(self):
        self._logger.log({"images": {"tag": "images_torch",
                                     "img_tensor":
                                         torch.from_numpy(self._npy_imgs)}})

    def test_imgs_npy(self):
        self._logger.log({"imgs": {"tag": "imgs_npy",
                                   "img_tensor": self._npy_imgs}})

    @unittest.skipIf(torch is None, "Torch Backend not installed")
    def test_imgs_torch(self):
        self._logger.log({"imgs": {"tag": "imgs_torch",
                                   "img_tensor":
                                       torch.from_numpy(self._npy_imgs)}})

    def test_pictures_npy(self):
        self._logger.log({"pictures": {"tag": "pictures_npy",
                                       "img_tensor": self._npy_imgs}})

    @unittest.skipIf(torch is None, "Torch Backend not installed")
    def test_pictures_torch(self):
        self._logger.log({"pictures": {"tag": "pictures_torch",
                                       "img_tensor":
                                           torch.from_numpy(self._npy_imgs)}})

    def test_image_with_boxes_npy(self):
        self._logger.log({"image_with_boxes": {
            "tag": "image_with_boxes_npy",
            "img_tensor": self._npy_imgs[0],
            "box_tensor": self._boxes_npy
        }})

    @unittest.skipIf(torch is None, "Torch Backend not installed")
    def test_image_with_boxes_torch(self):
        self._logger.log({"image_with_boxes": {
            "tag": "image_with_boxes_torch",
            "img_tensor": torch.from_numpy(self._npy_imgs[0]),
            "box_tensor": torch.from_numpy(self._boxes_npy)
        }})

    def test_bounding_boxes_npy(self):
        self._logger.log({"bounding_boxes": {
            "tag": "bounding_boxes_torch",
            "img_tensor": self._npy_imgs[0],
            "box_tensor": self._boxes_npy
        }})

    @unittest.skipIf(torch is None, "Torch Backend not installed")
    def test_bounding_boxes_torch(self):

        self._logger.log({"bounding_boxes": {
            "tag": "bounding_boxes_torch",
            "img_tensor": torch.from_numpy(self._npy_imgs[0]),
            "box_tensor": torch.from_numpy(self._boxes_npy)
        }})

    def test_bboxes_npy(self):
        self._logger.log({"bboxes": {
            "tag": "bboxes_npy",
            "img_tensor": self._npy_imgs[0],
            "box_tensor": self._boxes_npy
        }})

    @unittest.skipIf(torch is None, "Torch Backend not installed")
    def test_bboxes_torch(self):
        self._logger.log({"bboxes": {
            "tag": "bboxes_torch",
            "img_tensor": torch.from_numpy(self._npy_imgs[0]),
            "box_tensor": torch.from_numpy(self._boxes_npy)
        }})

    def test_scalar(self):
        for _scalar in self._scalars:
            self._logger.log({
                "scalar": {
                    "tag": "scalar",
                    "scalar_value": _scalar["1"]
                }
            })

    def test_scalar_npy(self):
        for _scalar in self._scalars:
            self._logger.log({
                "scalar": {
                    "tag": "scalar_npy",
                    "scalar_value": np.array(_scalar["1"])
                }
            })

    @unittest.skipIf(torch is None, "Torch Backend not installed")
    def test_scalar_torch(self):
        for _scalar in self._scalars:
            self._logger.log({
                "scalar": {
                    "tag": "scalar_torch",
                    "scalar_value": torch.tensor(_scalar["1"])
                }
            })

    def test_value(self):
        for _scalar in self._scalars:
            self._logger.log({
                "value": {
                    "tag": "value",
                    "scalar_value": _scalar["1"]
                }
            })

    def test_value_npy(self):
        for _scalar in self._scalars:
            self._logger.log({
                "value": {
                    "tag": "value_npy",
                    "scalar_value": np.array(_scalar["1"])
                }
            })

    @unittest.skipIf(torch is None, "Torch Backend not installed")
    def test_value_torch(self):
        for _scalar in self._scalars:
            self._logger.log({
                "value": {
                    "tag": "value_torch",
                    "scalar_value": torch.tensor(_scalar["1"])
                }
            })

    def test_scalars(self):
        for _scalar in self._scalars:
            self._logger.log({
                "scalars": {
                    "main_tag": "scalars",
                    "tag_scalar_dict": _scalar
                }
            })

    def test_scalars_npy(self):
        for _scalar in self._scalars:
            self._logger.log({
                "scalars": {
                    "main_tag": "scalars_npy",
                    "tag_scalar_dict": {k: np.array(v)
                                        for k, v in _scalar.items()}
                }
            })

    @unittest.skipIf(torch is None, "Torch Backend not installed")
    def test_scalars_torch(self):
        for _scalar in self._scalars:
            self._logger.log({
                "scalars": {
                    "main_tag": "scalars_torch",
                    "tag_scalar_dict": {k: torch.tensor(v)
                                        for k, v in _scalar.items()}
                }
            })

    def test_values(self):
        for _scalar in self._scalars:
            self._logger.log({
                "values": {
                    "main_tag": "values",
                    "tag_scalar_dict": _scalar
                }
            })

    def test_values_npy(self):
        for _scalar in self._scalars:
            self._logger.log({
                "values": {
                    "main_tag": "values_npy",
                    "tag_scalar_dict": {k: np.array(v)
                                        for k, v in _scalar.items()}
                }
            })

    @unittest.skipIf(torch is None, "Torch Backend not installed")
    def test_values_torch(self):
        for _scalar in self._scalars:
            self._logger.log({
                "values": {
                    "main_tag": "values_torch",
                    "tag_scalar_dict": {k: torch.tensor(v)
                                        for k, v in _scalar.items()}
                }
            })

    def test_histogram_npy(self):
        self._logger.log({
            "histogram": {
                "tag": "histogram_npy",
                "values": self._hist_vals
            }
        })

    @unittest.skipIf(torch is None, "Torch Backend not installed")
    def test_histogram_torch(self):
        self._logger.log({
            "histogram": {
                "tag": "histogram_torch",
                "values": torch.from_numpy(self._hist_vals)
            }
        })

    def test_hist_npy(self):
        self._logger.log({
            "hist": {
                "tag": "hist_npy",
                "values": self._hist_vals
            }
        })

    @unittest.skipIf(torch is None, "Torch Backend not installed")
    def test_hist_torch(self):
        self._logger.log({
            "hist": {
                "tag": "hist_torch",
                "values": torch.from_numpy(self._hist_vals)
            }
        })

    def test_figure(self):
        from matplotlib.pyplot import figure, imshow, close
        _fig = figure()
        imshow(self._npy_imgs[0][0])
        self._logger.log({
            "figure": {
                "tag": "figure",
                "figure": _fig
            }
        })
        close()

    def test_fig(self):
        from matplotlib.pyplot import figure, imshow, close
        _fig = figure()
        imshow(self._npy_imgs[0][0])
        self._logger.log({
            "fig": {
                "tag": "fig",
                "figure": _fig
            }
        })
        close()

    def test_audio_npy(self):
        self._logger.log({"audio": {
            "tag": "audio_npy",
            "snd_tensor": self._audio_sample_npy
        }})
    #
    @unittest.skipIf(torch is None, "Torch Backend not installed")
    def test_audio_torch(self):
        self._logger.log({"audio": {
            "tag": "audio_torch",
            "snd_tensor": torch.from_numpy(self._audio_sample_npy)
        }})

    def test_sound_npy(self):
        self._logger.log({"sound": {
            "tag": "sound_npy",
            "snd_tensor": self._audio_sample_npy
        }})

    @unittest.skipIf(torch is None, "Torch Backend not installed")
    def test_sound_torch(self):
        self._logger.log({"sound": {
            "tag": "sound_torch",
            "snd_tensor": torch.from_numpy(self._audio_sample_npy)
        }})

    def test_video_npy(self):
        # add channel and batch dimension for format BTCHW
        vid = self._npy_imgs.reshape((1, *self._npy_imgs.shape))
        print(vid.shape)

        self._logger.log({"video": {
            "tag": "video_npy",
            "vid_tensor": vid,
            "fps": 1
        }})

    @unittest.skipIf(torch is None, "Torch Backend not installed")
    def test_video_torch(self):
        # add channel and batch dimension for format BTCHW
        vid = self._npy_imgs.reshape((1, *self._npy_imgs.shape))

        self._logger.log({"video": {
            "tag": "video_torch",
            "vid_tensor": torch.from_numpy(vid),
            "fps": 1
        }})

    def test_text(self):
        self._logger.log({"text": {
            "tag": "text",
            "text_string": self._text_string
        }})

    @unittest.skipIf(tf is None, "TF Backend not installed")
    def test_graph_tf(self):
        self._logger.log({"graph_tf": {
            "graph": self._model_tf._graph
        }})

    @unittest.skipIf(torch is None, "Torch Backend not installed")
    def test_graph_torch(self):

        input_tensor = self._npy_imgs[0]
        input_tensor = input_tensor.reshape(1, *input_tensor.shape)

        self._logger.log({
            "graph_pytorch": {
                "model": self._model_torch,
                "input_to_model": torch.from_numpy(input_tensor).float()
            }
        })

    @unittest.skipIf(torch is None, "Torch Backend not installed")
    @unittest.skipIf(onnx is None, reason="ONNX not installed")
    def test_graph_onnx(self):
        import os
        input_tensor = self._npy_imgs[0]
        input_tensor = input_tensor.reshape(1, *input_tensor.shape)
        torch.onnx.export(self._model_torch,
                          torch.from_numpy(input_tensor).float(),
                          os.path.abspath("model.onnx"))
        self._logger.log({
            "graph_onnx": {"prototxt": os.path.abspath("model.onnx")}
        })

    def test_embedding_npy(self):
        self._logger.log({"embedding": {
            "mat": self._embedding_npy
        }})

    @unittest.skipIf(torch is None, "Torch Backend not installed")
    def test_embedding_torch(self):
        self._logger.log({"embedding": {
            "mat": torch.from_numpy(self._embedding_npy)
        }})

    def test_pr_curve_npy(self):
        self._logger.log({"pr_curve": {
            "tag": "pr_curve_npy",
            "labels": self._labels_npy,
            "predictions": self._predictions_npy
        }})

    @unittest.skipIf(torch is None, "Torch Backend not installed")
    def test_pr_curve_torch(self):
        self._logger.log({"pr_curve": {
            "tag": "pr_curve_torch",
            "labels": torch.from_numpy(self._labels_npy),
            "predictions": torch.from_numpy(self._predictions_npy)
        }})

    def test_pr_npy(self):
        self._logger.log({"pr": {
            "tag": "pr_npy",
            "labels": self._labels_npy,
            "predictions": self._predictions_npy
        }})

    @unittest.skipIf(torch is None, "Torch Backend not installed")
    def test_pr_torch(self):
        self._logger.log({"pr": {
            "tag": "pr_torch",
            "labels": torch.from_numpy(self._labels_npy),
            "predictions": torch.from_numpy(self._predictions_npy)
        }})

    def tearDown(self) -> None:
        self._destroy_logger(self._logger)
        self._logger = None


if __name__ == '__main__':
    from multiprocessing import freeze_support
    freeze_support()
    unittest.main()
