import importlib

import cvbase
from cvbase.image.backend_base import ImageBase, ImageDrawBase

_backend = importlib.import_module('.backend_' + cvbase.get_backend(),
                                   'cvbase.image')

Image = _backend.Image
ImageDraw = _backend.ImageDraw
