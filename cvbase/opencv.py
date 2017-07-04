import cv2


def use_opencv3():
    return cv2.__version__.split('.')[0] == '3'


USE_OPENCV3 = use_opencv3()

if USE_OPENCV3:
    from cv2 import IMREAD_COLOR, IMREAD_GRAYSCALE, IMREAD_UNCHANGED
    from cv2 import INTER_MAX, WARP_FILL_OUTLIERS, WARP_INVERSE_MAP
else:
    from cv2 import CV_LOAD_IMAGE_COLOR as IMREAD_COLOR
    from cv2 import CV_LOAD_IMAGE_GRAYSCALE as IMREAD_GRAYSCALE
    from cv2 import CV_LOAD_IMAGE_UNCHANGED as IMREAD_UNCHANGED

from cv2 import (INTER_NEAREST, INTER_LINEAR, INTER_CUBIC, INTER_AREA,
                 INTER_LANCZOS4)
