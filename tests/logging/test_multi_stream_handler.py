from delira.logging import MultiStreamHandler

import logging
import numpy as np
import os
import sys

def test_logger():
    handler = MultiStreamHandler(sys.stdout, sys.stderr)

    logging.basicConfig(level=logging.INFO, 
        handlers=[handler])

    logger = logging.getLogger(__name__)
    logger.info("Test Info for Unittest")

