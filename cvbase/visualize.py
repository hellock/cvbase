from enum import Enum

import cv2
import numpy as np

from cvbase.image import read_img


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
        cv2.imshow(win_name, img)
        cv2.waitKey(wait_time)
    if out_file is not None:
        cv2.imwrite(out_file, img)


def voc_labels():
    return [
        'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat',
        'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person',
        'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'
    ]


def det_labels():
    return [
        'accordion', 'airplane', 'ant', 'antelope', 'apple', 'armadillo',
        'artichoke', 'axe', 'baby_bed', 'backpack', 'bagel', 'balance_beam',
        'banana', 'band_aid', 'banjo', 'baseball', 'basketball', 'bathing_cap',
        'beaker', 'bear', 'bee', 'bell_pepper', 'bench', 'bicycle', 'binder',
        'bird', 'bookshelf', 'bow_tie', 'bow', 'bowl', 'brassiere', 'burrito',
        'bus', 'butterfly', 'camel', 'can_opener', 'car', 'cart', 'cattle',
        'cello', 'centipede', 'chain_saw', 'chair', 'chime', 'cocktail_shaker',
        'coffee_maker', 'computer_keyboard', 'computer_mouse', 'corkscrew',
        'cream', 'croquet_ball', 'crutch', 'cucumber', 'cup_or_mug', 'diaper',
        'digital_clock', 'dishwasher', 'dog', 'domestic_cat', 'dragonfly',
        'drum', 'dumbbell', 'electric_fan', 'elephant', 'face_powder', 'fig',
        'filing_cabinet', 'flower_pot', 'flute', 'fox', 'french_horn', 'frog',
        'frying_pan', 'giant_panda', 'goldfish', 'golf_ball', 'golfcart',
        'guacamole', 'guitar', 'hair_dryer', 'hair_spray', 'hamburger',
        'hammer', 'hamster', 'harmonica', 'harp', 'hat_with_a_wide_brim',
        'head_cabbage', 'helmet', 'hippopotamus', 'horizontal_bar', 'horse',
        'hotdog', 'iPod', 'isopod', 'jellyfish', 'koala_bear', 'ladle',
        'ladybug', 'lamp', 'laptop', 'lemon', 'lion', 'lipstick', 'lizard',
        'lobster', 'maillot', 'maraca', 'microphone', 'microwave', 'milk_can',
        'miniskirt', 'monkey', 'motorcycle', 'mushroom', 'nail', 'neck_brace',
        'oboe', 'orange', 'otter', 'pencil_box', 'pencil_sharpener', 'perfume',
        'person', 'piano', 'pineapple', 'ping-pong_ball', 'pitcher', 'pizza',
        'plastic_bag', 'plate_rack', 'pomegranate', 'popsicle', 'porcupine',
        'power_drill', 'pretzel', 'printer', 'puck', 'punching_bag', 'purse',
        'rabbit', 'racket', 'ray', 'red_panda', 'refrigerator',
        'remote_control', 'rubber_eraser', 'rugby_ball', 'ruler',
        'salt_or_pepper_shaker', 'saxophone', 'scorpion', 'screwdriver',
        'seal', 'sheep', 'ski', 'skunk', 'snail', 'snake', 'snowmobile',
        'snowplow', 'soap_dispenser', 'soccer_ball', 'sofa', 'spatula',
        'squirrel', 'starfish', 'stethoscope', 'stove', 'strainer',
        'strawberry', 'stretcher', 'sunglasses', 'swimming_trunks', 'swine',
        'syringe', 'table', 'tape_player', 'tennis_ball', 'tick', 'tie',
        'tiger', 'toaster', 'traffic_light', 'train', 'trombone', 'trumpet',
        'turtle', 'tv_or_monitor', 'unicycle', 'vacuum', 'violin',
        'volleyball', 'waffle_iron', 'washer', 'water_bottle', 'watercraft',
        'whale', 'wine_bottle', 'zebra'
    ]


def vid_labels():
    return [
        'airplane', 'antelope', 'bear', 'bicycle', 'bird', 'bus', 'car',
        'cattle', 'dog', 'domestic_cat', 'elephant', 'fox', 'giant_panda',
        'hamster', 'horse', 'lion', 'lizard', 'monkey', 'motorcycle', 'rabbit',
        'red_panda', 'sheep', 'snake', 'squirrel', 'tiger', 'train', 'turtle',
        'watercraft', 'whale', 'zebra'
    ]


def read_labels(file_or_labels):
    """Read labels from file or list"""
    if file_or_labels in ['voc', 'det', 'vid']:
        return eval(file_or_labels + '_labels()')
    if isinstance(file_or_labels, list):
        label_names = file_or_labels
    elif isinstance(file_or_labels, str):
        with open(file_or_labels, 'r') as fin:
            label_names = [line.rstrip('\n') for line in fin]
    return label_names


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
            cv2.putText(img, label_text,
                        (bboxes_int[j, 0], bboxes_int[j, 1] - 2),
                        cv2.FONT_HERSHEY_COMPLEX, font_scale, text_color)
    if show:
        cv2.imshow(win_name, img)
        cv2.waitKey(wait_time)
    if out_file is not None:
        cv2.imwrite(out_file, img)
