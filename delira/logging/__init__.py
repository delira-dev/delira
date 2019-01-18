
import os
if "torch" in os.environ["DELIRA_BACKEND"]:
    from .deprecated import VisdomStreamHandler, VisdomImageSaveStreamHandler, \
        VisdomImageSaveHandler, VisdomImageHandler, ImgSaveHandler

from .multistream_handler import MultiStreamHandler
from .trixi_handler import TrixiHandler
