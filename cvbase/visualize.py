from enum import Enum

import cv2
import numpy as np

from cvbase.det.labels import read_labels
from cvbase.image import read_img, write_img


class Color(Enum):
    """Color associated with RGB values

    8 colors in total: red, green, blue, cyan, yellow, magenta, white and black.
    """
    red = (0, 0, 255)
    green = (0, 255, 0)
    blue = (255, 0, 0)
    cyan = (255, 255, 0)
    yellow = (0, 255, 255)
    magenta = (255, 0, 255)
    white = (255, 255, 255)
    black = (0, 0, 0)


def show_img(img, win_name='', wait_time=0):
    """Show an image

    Args:
        img(str or ndarray): the image to be shown
        win_name(str): the window name
        wait_time(int): value of waitKey param
    """
    cv2.imshow(win_name, read_img(img))
    cv2.waitKey(wait_time)


def draw_bboxes(img, bboxes, colors=Color.green, top_k=0, thickness=1,
                show=True, win_name='', wait_time=0, out_file=None):  # yapf: disable
    """Draw bboxes on an image

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
        _bboxes = _bboxes.astype(np.int32)
        if top_k <= 0:
            _top_k = _bboxes.shape[0]
        else:
            _top_k = min(top_k, _bboxes.shape[0])
        for j in range(_top_k):
            left_top = (_bboxes[j, 0], _bboxes[j, 1])
            right_bottom = (_bboxes[j, 2], _bboxes[j, 3])
            cv2.rectangle(
                img, left_top, right_bottom, colors[i], thickness=thickness)
    if show:
        show_img(img, win_name, wait_time)
    if out_file is not None:
        write_img(img, out_file)


def draw_bboxes_with_label(img, bboxes, labels, top_k=0, bbox_color=Color.green,
                           text_color=Color.green, thickness=1, font_scale=0.5,
                           show=True, win_name='', wait_time=0, out_file=None):  # yapf: disable
    """Draw bboxes with label text in image

    Args:
        img(str or ndarray): the image to be shown
        bboxes(list or ndarray): a list of ndarray of shape (k, 4)
        labels(str or list): label name file or list of label names
        top_k(int): draw top_k bboxes only if positive
        bbox_color(Color or tuple): color to draw bboxes
        text_color(Color or tuple): color to draw label texts
        thickness(int): thickness of bbox lines
        font_scale(float): font scales
        show(bool): whether to show the image
        win_name(str): the window name
        wait_time(int): value of waitKey param
        out_file(str or None): the filename to write the image
    """
    img = read_img(img)
    label_names = read_labels(labels)
    if isinstance(bbox_color, Color):
        bbox_color = bbox_color.value
    if isinstance(text_color, Color):
        text_color = text_color.value
    assert len(bboxes) == len(label_names)
    for i, _bboxes in enumerate(bboxes):
        bboxes_int = _bboxes[:, :4].astype(np.int32)
        if top_k <= 0:
            _top_k = _bboxes.shape[0]
        else:
            _top_k = min(top_k, _bboxes.shape[0])
        for j in range(_top_k):
            left_top = (bboxes_int[j, 0], bboxes_int[j, 1])
            right_bottom = (bboxes_int[j, 2], bboxes_int[j, 3])
            cv2.rectangle(
                img, left_top, right_bottom, bbox_color, thickness=thickness)
            if _bboxes.shape[1] > 4:
                label_text = '{}|{:.02f}'.format(label_names[i], _bboxes[j, 4])
            else:
                label_text = label_names[i]
            cv2.putText(img, label_text, (bboxes_int[j, 0],
                                          bboxes_int[j, 1] - 2),
                        cv2.FONT_HERSHEY_COMPLEX, font_scale, text_color)
    if show:
        show_img(img, win_name, wait_time)
    if out_file is not None:
        write_img(img, out_file)
