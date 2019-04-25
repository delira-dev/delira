from delira.logging import MultiStreamHandler

import logging
import numpy as np
import os
import sys

import unittest


class MultiStreamHandlerTest(unittest.TestCase):

    def test_logger(self):
        handler = MultiStreamHandler(sys.stdout, sys.stderr)

        logging.basicConfig(level=logging.INFO,
                            handlers=[handler])

        logger = logging.getLogger(__name__)

        with self.assertLogs(__name__, level='INFO'):
            logger.info("Test Info for Unittest")
