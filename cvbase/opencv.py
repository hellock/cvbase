import cv2


def use_opencv3():
    return cv2.__version__.split('.')[0] == '3'


USE_OPENCV3 = use_opencv3()
