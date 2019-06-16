import logging
import sys
import unittest

from delira.logging import MultiStreamHandler
from ..utils import check_for_no_backend


class MultiStreamHandlerTest(unittest.TestCase):

    @unittest.skipUnless(check_for_no_backend(),
                         "Test should be only executed if no backend is "
                         "installed and specified")
    def test_logger(self):
        handler = MultiStreamHandler(sys.stdout, sys.stderr)

        logging.basicConfig(level=logging.INFO,
                            handlers=[handler])

        logger = logging.getLogger(__name__)

        with self.assertLogs(__name__, level='INFO'):
            logger.info("Test Info for Unittest")


if __name__ == '__main__':
    unittest.main()
