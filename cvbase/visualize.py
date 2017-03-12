from enum import Enum

import cv2
import numpy as np

from cvbase.io import read_img


class Color(Enum):
    red = (0, 0, 255)
    green = (0, 255, 0)
    blue = (255, 0, 0)
    cyan = (255, 255, 0)
    yellow = (0, 255, 255)
    magenta = (255, 0, 255)
    white = (255, 255, 255)
    black = (0, 0, 0)


def draw_bboxes(img, bboxes, colors=Color.green, top_k=0, thickness=1,
                show=True, win_name='', wait_time=0, out_file=None):  # yapf: disable
    """Draw bboxes in image
    Args:
        img(str or ndarray): the image to be shown
        bboxes(list or ndarray): a list of ndarray of shape (k, 4)
        colors(list or Color or tuple): a list of colors, corresponding to bboxes
        top_k(int): draw top_k bboxes only if positive
        thickness(int): thickness of lines
        show(bool): whether to show the image
        win_name(str): the window name
        wait_time(int): value of waitKey param
        out_file(str or None): the filename to write the image
    """
    img = read_img(img)
    if isinstance(bboxes, np.ndarray):
        bboxes = [bboxes]
    if isinstance(colors, (tuple, Color)):
        colors = [colors for _ in range(len(bboxes))]
    for i in range(len(colors)):
        if isinstance(colors[i], Color):
            colors[i] = colors[i].value
    assert len(bboxes) == len(colors)

    for i, _bboxes in enumerate(bboxes):
        if top_k <= 0:
            _top_k = _bboxes.shape[0]
        else:
            _top_k = min(top_k, _bboxes.shape[0])
        for j in range(_top_k):
            cv2.rectangle(img, (_bboxes[j, 0], _bboxes[j, 1]),
                          (_bboxes[j, 2], _bboxes[j, 3]), colors[i])
    if show:
        cv2.imshow(win_name, img)
        cv2.waitKey(wait_time)
    if out_file is not None:
        cv2.imwrite(out_file, img)
