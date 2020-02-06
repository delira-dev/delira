from delira.logging import Logger, TensorboardBackend, make_logger

from tests.utils import check_for_torch_backend, check_for_tf_graph_backend

import unittest

try:
    import tensorflow as tf
except ImportError:
    tf = None

try:
    import torch
except ImportError:
    torch = None

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
            tf.reset_default_graph()
            input = np.zeros(shape=(1, 28, 28, 1))

            layers = tf.keras.layers
            self._model_tf = tf.keras.Sequential(
                [layers.Conv2D(
                    32,
                    5,
                    padding='same',
                    data_format="channels_last",
                    activation=tf.nn.relu),
                    layers.Conv2D(
                        64,
                        5,
                        padding='same',
                        data_format="channels_last",
                        activation=tf.nn.relu),
                 ])
            self._model_tf.build(input_shape=input.shape)

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
            {"logdir": os.path.join(".", "runs", self._testMethodName)}
        ))

    def _check_for_tag(self, tag, logdir=None):

        if logdir is None:
            try:
                logdir = self._logger._backend._writer.logdir
            except AttributeError:
                logdir = self._logger._backend._writer.log_dir

        file = [os.path.join(logdir, x)
                for x in os.listdir(logdir)
                if os.path.isfile(os.path.join(logdir, x))][0]

        if tf is not None:
            ret_val = False
            for e in tf.train.summary_iterator(file):
                for v in e.summary.value:
                    if v.tag == tag:
                        ret_val = True
                        break
                if ret_val:
                    break

            self.assertTrue(ret_val)

    @staticmethod
    def _destroy_logger(logger: Logger):
        logger.close()
        del logger
        gc.collect()

    def test_image_npy(self):
        self._logger.log({"image": {"tag": "image_npy",
                                    "img_tensor": self._npy_imgs[0]}})
        self._check_for_tag("image_npy")

    @unittest.skipUnless(check_for_torch_backend(),
                         "Torch Backend not installed")
    def test_image_torch(self):
        self._logger.log({"image": {"tag": "image_torch",
                                    "img_tensor":
                                        torch.from_numpy(self._npy_imgs[0])}})
        self._check_for_tag("image_torch")

    def test_img_npy(self):
        self._logger.log({"img": {"tag": "img_npy",
                                  "img_tensor": self._npy_imgs[0]}})
        self._check_for_tag("img_npy")

    @unittest.skipUnless(check_for_torch_backend(),
                         "Torch Backend not installed")
    def test_img_torch(self):
        self._logger.log({"img": {"tag": "img_torch",
                                  "img_tensor":
                                      torch.from_numpy(self._npy_imgs[0])}})
        self._check_for_tag("img_torch")

    def test_picture_npy(self):
        self._logger.log({"picture": {"tag": "picture_npy",
                                      "img_tensor": self._npy_imgs[0]}})
        self._check_for_tag("picture_npy")

    @unittest.skipUnless(check_for_torch_backend(),
                         "Torch Backend not installed")
    def test_picture_torch(self):
        self._logger.log({
            "picture": {
                "tag": "picture_torch",
                "img_tensor": torch.from_numpy(self._npy_imgs[0])}})
        self._check_for_tag("picture_torch")

    def test_images_npy(self):
        self._logger.log({"images": {"tag": "images_npy",
                                     "img_tensor": self._npy_imgs}})
        self._check_for_tag("images_npy")

    @unittest.skipUnless(check_for_torch_backend(),
                         "Torch Backend not installed")
    def test_images_torch(self):
        self._logger.log({"images": {"tag": "images_torch",
                                     "img_tensor":
                                         torch.from_numpy(self._npy_imgs)}})
        self._check_for_tag("images_torch")

    def test_imgs_npy(self):
        self._logger.log({"imgs": {"tag": "imgs_npy",
                                   "img_tensor": self._npy_imgs}})
        self._check_for_tag("imgs_npy")

    @unittest.skipUnless(check_for_torch_backend(),
                         "Torch Backend not installed")
    def test_imgs_torch(self):
        self._logger.log({"imgs": {"tag": "imgs_torch",
                                   "img_tensor":
                                       torch.from_numpy(self._npy_imgs)}})
        self._check_for_tag("imgs_torch")

    def test_pictures_npy(self):
        self._logger.log({"pictures": {"tag": "pictures_npy",
                                       "img_tensor": self._npy_imgs}})
        self._check_for_tag("pictures_npy")

    @unittest.skipUnless(check_for_torch_backend(),
                         "Torch Backend not installed")
    def test_pictures_torch(self):
        self._logger.log({"pictures": {"tag": "pictures_torch",
                                       "img_tensor":
                                           torch.from_numpy(self._npy_imgs)}})
        self._check_for_tag("pictures_torch")

    def test_image_with_boxes_npy(self):
        self._logger.log({"image_with_boxes": {
            "tag": "image_with_boxes_npy",
            "img_tensor": self._npy_imgs[0],
            "box_tensor": self._boxes_npy
        }})
        self._check_for_tag("image_with_boxes_npy")

    @unittest.skipUnless(check_for_torch_backend(),
                         "Torch Backend not installed")
    def test_image_with_boxes_torch(self):
        self._logger.log({"image_with_boxes": {
            "tag": "image_with_boxes_torch",
            "img_tensor": torch.from_numpy(self._npy_imgs[0]),
            "box_tensor": torch.from_numpy(self._boxes_npy)
        }})
        self._check_for_tag("image_with_boxes_torch")

    def test_bounding_boxes_npy(self):
        self._logger.log({"bounding_boxes": {
            "tag": "bounding_boxes_npy",
            "img_tensor": self._npy_imgs[0],
            "box_tensor": self._boxes_npy
        }})
        self._check_for_tag("bounding_boxes_npy")

    @unittest.skipUnless(check_for_torch_backend(),
                         "Torch Backend not installed")
    def test_bounding_boxes_torch(self):

        self._logger.log({"bounding_boxes": {
            "tag": "bounding_boxes_torch",
            "img_tensor": torch.from_numpy(self._npy_imgs[0]),
            "box_tensor": torch.from_numpy(self._boxes_npy)
        }})
        self._check_for_tag("bounding_boxes_torch")

    def test_bboxes_npy(self):
        self._logger.log({"bboxes": {
            "tag": "bboxes_npy",
            "img_tensor": self._npy_imgs[0],
            "box_tensor": self._boxes_npy
        }})
        self._check_for_tag("bboxes_npy")

    @unittest.skipUnless(check_for_torch_backend(),
                         "Torch Backend not installed")
    def test_bboxes_torch(self):
        self._logger.log({"bboxes": {
            "tag": "bboxes_torch",
            "img_tensor": torch.from_numpy(self._npy_imgs[0]),
            "box_tensor": torch.from_numpy(self._boxes_npy)
        }})
        self._check_for_tag("bboxes_torch")

    def test_scalar(self):
        for _scalar in self._scalars:
            self._logger.log({
                "scalar": {
                    "tag": "scalar",
                    "scalar_value": _scalar["1"]
                }
            })
        self._check_for_tag("scalar")

    def test_scalar_npy(self):
        for _scalar in self._scalars:
            self._logger.log({
                "scalar": {
                    "tag": "scalar_npy",
                    "scalar_value": np.array(_scalar["1"])
                }
            })

        self._check_for_tag("scalar_npy")

    @unittest.skipUnless(check_for_torch_backend(),
                         "Torch Backend not installed")
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
        self._check_for_tag("value")

    def test_value_npy(self):
        for _scalar in self._scalars:
            self._logger.log({
                "value": {
                    "tag": "value_npy",
                    "scalar_value": np.array(_scalar["1"])
                }
            })
        self._check_for_tag("value_npy")

    @unittest.skipUnless(check_for_torch_backend(),
                         "Torch Backend not installed")
    def test_value_torch(self):
        for _scalar in self._scalars:
            self._logger.log({
                "value": {
                    "tag": "value_torch",
                    "scalar_value": torch.tensor(_scalar["1"])
                }
            })
        self._check_for_tag("value_torch")

    def test_scalars(self):
        for _scalar in self._scalars:
            self._logger.log({
                "scalars": {
                    "main_tag": "scalars",
                    "tag_scalar_dict": _scalar,
                    "sep": "/"
                }
            })

        for k in self._scalars[0].keys():
            self._check_for_tag("scalars/" + k)

    def test_scalars_npy(self):
        for _scalar in self._scalars:
            self._logger.log({
                "scalars": {
                    "main_tag": "scalars_npy",
                    "tag_scalar_dict": {k: np.array(v)
                                        for k, v in _scalar.items()},
                    "sep": "/"
                }
            })

        for k in self._scalars[0].keys():
            self._check_for_tag("scalars_npy/" + k)

    @unittest.skipUnless(check_for_torch_backend(),
                         "Torch Backend not installed")
    def test_scalars_torch(self):
        for _scalar in self._scalars:
            self._logger.log({
                "scalars": {
                    "main_tag": "scalars_torch",
                    "tag_scalar_dict": {k: torch.tensor(v)
                                        for k, v in _scalar.items()},
                    "sep": "/"
                }
            })

        for k in self._scalars[0].keys():
            self._check_for_tag("scalars_torch/" + k)

    def test_values(self):
        for _scalar in self._scalars:
            self._logger.log({
                "values": {
                    "main_tag": "values",
                    "tag_scalar_dict": _scalar,
                    "sep": "/"
                }
            })

        for k in self._scalars[0].keys():
            self._check_for_tag("values/" + k)

    def test_values_npy(self):
        for _scalar in self._scalars:
            self._logger.log({
                "values": {
                    "main_tag": "values_npy",
                    "tag_scalar_dict": {k: np.array(v)
                                        for k, v in _scalar.items()},
                    "sep": "/"
                }
            })

        for k in self._scalars[0].keys():
            self._check_for_tag("values_npy/" + k)

    @unittest.skipUnless(check_for_torch_backend(),
                         "Torch Backend not installed")
    def test_values_torch(self):
        for _scalar in self._scalars:
            self._logger.log({
                "values": {
                    "main_tag": "values_torch",
                    "tag_scalar_dict": {k: torch.tensor(v)
                                        for k, v in _scalar.items()},
                    "sep": "/"
                }
            })

        for k in self._scalars[0].keys():
            self._check_for_tag("values_torch/" + k)

    def test_histogram_npy(self):
        self._logger.log({
            "histogram": {
                "tag": "histogram_npy",
                "values": self._hist_vals
            }
        })

        self._check_for_tag("histogram_npy")

    @unittest.skipUnless(check_for_torch_backend(),
                         "Torch Backend not installed")
    def test_histogram_torch(self):
        self._logger.log({
            "histogram": {
                "tag": "histogram_torch",
                "values": torch.from_numpy(self._hist_vals)
            }
        })

        self._check_for_tag("histogram_torch")

    def test_hist_npy(self):
        self._logger.log({
            "hist": {
                "tag": "hist_npy",
                "values": self._hist_vals
            }
        })

        self._check_for_tag("hist_npy")

    @unittest.skipUnless(check_for_torch_backend(),
                         "Torch Backend not installed")
    def test_hist_torch(self):
        self._logger.log({
            "hist": {
                "tag": "hist_torch",
                "values": torch.from_numpy(self._hist_vals)
            }
        })

        self._check_for_tag("hist_torch")

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

        self._check_for_tag("figure")

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

        self._check_for_tag("fig")

    def test_audio_npy(self):
        self._logger.log({"audio": {
            "tag": "audio_npy",
            "snd_tensor": self._audio_sample_npy
        }})

        self._check_for_tag("audio_npy")

    @unittest.skipUnless(check_for_torch_backend(),
                         "Torch Backend not installed")
    def test_audio_torch(self):
        self._logger.log({"audio": {
            "tag": "audio_torch",
            "snd_tensor": torch.from_numpy(self._audio_sample_npy)
        }})

        self._check_for_tag("audio_torch")

    def test_sound_npy(self):
        self._logger.log({"sound": {
            "tag": "sound_npy",
            "snd_tensor": self._audio_sample_npy
        }})

        self._check_for_tag("sound_npy")

    @unittest.skipUnless(check_for_torch_backend(),
                         "Torch Backend not installed")
    def test_sound_torch(self):
        self._logger.log({"sound": {
            "tag": "sound_torch",
            "snd_tensor": torch.from_numpy(self._audio_sample_npy)
        }})

        self._check_for_tag("sound_torch")

    def test_video_npy(self):
        # add channel and batch dimension for format BTCHW
        vid = self._npy_imgs.reshape((1, *self._npy_imgs.shape))

        self._logger.log({"video": {
            "tag": "video_npy",
            "vid_tensor": vid,
            "fps": 1
        }})
        self._check_for_tag("video_npy")

    @unittest.skipUnless(check_for_torch_backend(),
                         "Torch Backend not installed")
    def test_video_torch(self):
        # add channel and batch dimension for format BTCHW
        vid = self._npy_imgs.reshape((1, *self._npy_imgs.shape))

        self._logger.log({"video": {
            "tag": "video_torch",
            "vid_tensor": torch.from_numpy(vid),
            "fps": 1
        }})

        self._check_for_tag("video_torch")

    def test_text(self):
        self._logger.log({"text": {
            "tag": "text",
            "text_string": self._text_string
        }})

        self._check_for_tag("text/text_summary")

    @unittest.skipUnless(check_for_tf_graph_backend(),
                         "TF Backend not installed")
    def test_graph_tf(self):

        run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
        run_metadata = tf.RunMetadata()

        with tf.Session() as sess:
            outputs = self._model_tf(
                np.zeros(
                    shape=(
                        1,
                        28,
                        28,
                        1),
                    dtype=np.float32))
            sess.run(tf.initializers.global_variables())
            sess.run(outputs, options=run_options, run_metadata=run_metadata)

        self._logger.log({"graph_tf": {
            "graph": self._model_tf._graph.as_graph_def(add_shapes=True),
            "run_metadata": run_metadata
        }})

    @unittest.skipUnless(check_for_torch_backend(),
                         "Torch Backend not installed")
    def test_graph_torch(self):

        input_tensor = self._npy_imgs[0]
        input_tensor = input_tensor.reshape(1, *input_tensor.shape)

        self._logger.log({
            "graph_pytorch": {
                "model": self._model_torch,
                "input_to_model": torch.from_numpy(input_tensor).float()
            }
        })

    @unittest.skipUnless(check_for_torch_backend(),
                         "Torch Backend not installed")
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

    @unittest.skipUnless(check_for_torch_backend(),
                         "Torch Backend not installed")
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
        self._check_for_tag("pr_curve_npy")

    @unittest.skipUnless(check_for_torch_backend(),
                         "Torch Backend not installed")
    def test_pr_curve_torch(self):
        self._logger.log({"pr_curve": {
            "tag": "pr_curve_torch",
            "labels": torch.from_numpy(self._labels_npy),
            "predictions": torch.from_numpy(self._predictions_npy)
        }})
        self._check_for_tag("pr_curve_torch")

    def test_pr_npy(self):
        self._logger.log({"pr": {
            "tag": "pr_npy",
            "labels": self._labels_npy,
            "predictions": self._predictions_npy
        }})
        self._check_for_tag("pr_npy")

    @unittest.skipUnless(check_for_torch_backend(),
                         "Torch Backend not installed")
    def test_pr_torch(self):
        self._logger.log({"pr": {
            "tag": "pr_torch",
            "labels": torch.from_numpy(self._labels_npy),
            "predictions": torch.from_numpy(self._predictions_npy)
        }})
        self._check_for_tag("pr_torch")

    def tearDown(self) -> None:
        self._destroy_logger(self._logger)
        self._logger = None


if __name__ == '__main__':
    from multiprocessing import freeze_support
    freeze_support()
    unittest.main()
