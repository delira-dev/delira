import unittest

from batchgenerators.transforms import ZoomTransform, PadTransform, Compose
import numpy as np

try:
    import numba
except ImportError:
    numba = None


class NumbaTest(unittest.TestCase):
    def setUp(self) -> None:
        from delira.data_loading.numba_transform import NumbaTransform, \
            NumbaCompose
        self._basic_zoom_trafo = ZoomTransform(3)
        self._numba_zoom_trafo = NumbaTransform(ZoomTransform, zoom_factors=3)
        self._basic_pad_trafo = PadTransform(new_size=(30, 30))
        self._numba_pad_trafo = NumbaTransform(PadTransform,
                                               new_size=(30, 30))

        self._basic_compose_trafo = Compose([self._basic_pad_trafo,
                                             self._basic_zoom_trafo])
        self._numba_compose_trafo = NumbaCompose([self._basic_pad_trafo,
                                                  self._basic_zoom_trafo])

        self._input = {"data": np.random.rand(10, 1, 24, 24)}

    def compare_transform_outputs(self, transform, numba_transform):
        output_normal = transform(**self._input)["data"]
        output_numba = numba_transform(**self._input)["data"]

        # only check for same shapes, since numba might apply slightly
        # different interpolations
        self.assertTupleEqual(output_normal.shape, output_numba.shape)

    @unittest.skipIf(numba is None, "Numba must be imported successfully")
    def test_zoom(self):
        self.compare_transform_outputs(self._basic_zoom_trafo,
                                       self._numba_zoom_trafo)

    @unittest.skipIf(numba is None, "Numba must be imported successfully")
    def test_pad(self):
        self.compare_transform_outputs(self._basic_pad_trafo,
                                       self._numba_pad_trafo)

    @unittest.skipIf(numba is None, "Numba must be imported successfully")
    def test_compose(self):
        self.compare_transform_outputs(self._basic_compose_trafo,
                                       self._numba_compose_trafo)


if __name__ == '__main__':
    unittest.main()
