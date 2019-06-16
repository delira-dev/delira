import logging
import unittest

import numpy as np
from trixi.logger import NumpyPlotFileLogger

from delira.logging import TrixiHandler
from ..utils import check_for_no_backend


class TrixiHandlerTest(unittest.TestCase):

    @unittest.skipUnless(check_for_no_backend(),
                         "Test should be only executed if no backend is "
                         "installed and specified")
    def test_trixi_logger(self):
        handler = TrixiHandler(NumpyPlotFileLogger,
                               img_dir="./imgs", plot_dir="./plots")

        logging.basicConfig(level=logging.INFO,
                            handlers=[handler])

        logger = logging.getLogger(__name__)

        with self.assertLogs(__name__, level='INFO'):
            logger.info(
                {'image': {"image": np.random.rand(28, 28),
                           "name": "test_img"}})


if __name__ == '__main__':
    unittest.main()
