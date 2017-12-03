import importlib
import sys

from .conversion import *
from .det import *
from .io import *
from .progress import *
from .timer import *
from .version import __version__
# require opencv
try:
    from . import legacy
    from .opencv import *
    from .video import *
except ImportError:
    pass

backend = ''


def get_backend():
    return backend


def set_backend(name):
    global backend
    backend = name
    if 'cvbase.image' in sys.modules:
        if sys.version_info > (3, 4):
            importlib.reload(sys.modules['cvbase.image'])
        else:
            reload(sys.modules['cvbase.image'])


def opencv_installed():
    try:
        import cv2
    except ImportError:
        return False
    else:
        return True


def pillow_installed():
    try:
        import pillow
    except ImportError:
        return False
    else:
        return True


if opencv_installed():
    set_backend('opencv')
elif pillow_installed():
    set_backend('pillow')
else:
    raise ImportError('Neither opencv nor pillow is installed!')

import cvbase.image
from .visualize import *
