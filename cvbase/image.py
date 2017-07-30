import cv2
import numpy as np

from cvbase.io import check_file_exist
from cvbase.opencv import IMREAD_COLOR, INTER_LINEAR


def read_img(img_or_path, flag=IMREAD_COLOR):
    """Read an image

    Args:
        img_or_path(ndarray or str): either an image or path of an image

    Returns:
        ndarray: image array
    """
    if isinstance(img_or_path, np.ndarray):
        return img_or_path
    elif isinstance(img_or_path, str):
        check_file_exist(img_or_path,
                         'img file does not exist: {}'.format(img_or_path))
        return cv2.imread(img_or_path, flag)
    else:
        raise TypeError('"img" must be a numpy array or a filename')


def img_from_bytes(content, flag=IMREAD_COLOR):
    """Read an image from bytes"""
    img_np = np.fromstring(content, np.uint8)
    img = cv2.imdecode(img_np, flag)
    return img


def write_img(img, file_path, params=None):
    return cv2.imwrite(file_path, img, params)


def scale_size(size, scale):
    w, h = size
    return int(w * float(scale) + 0.5), int(h * float(scale) + 0.5)


def resize(img, size, return_scale=False, interpolation=INTER_LINEAR):
    """Resize image by expected size

    Args:
        img(ndarray): image or image path
        size(tuple): (w, h)
        return_scale(bool): whether to return w_scale and h_scale
        interpolation(enum): interpolation method

    Returns:
        ndarray: resized image
    """
    img = read_img(img)
    h, w = img.shape[:2]
    resized_img = cv2.resize(img, size, interpolation=interpolation)
    if not return_scale:
        return resized_img
    else:
        w_scale = size[0] / float(w)
        h_scale = size[1] / float(h)
        return resized_img, w_scale, h_scale


def resize_like(img, dst_img, return_scale=False, interpolation=INTER_LINEAR):
    """Resize image to the same size of a given image

    Args:
        img(ndarray): image or image path
        dst_img(ndarray): the given image with expected size
        return_scale(bool): whether to return w_scale and h_scale
        interpolation(enum): interpolation method

    Returns:
        ndarray: resized image
    """
    h, w = dst_img.shape[:2]
    return resize(img, (w, h), return_scale, interpolation)


def resize_by_ratio(img, ratio, interpolation=INTER_LINEAR):
    """Resize image by a ratio

    Args:
        img(ndarray): image or image path
        ratio(float): scale factor
        interpolation(enum): interpolation method

    Returns:
        ndarray: resized image
    """
    assert isinstance(ratio, (float, int)) and ratio > 0
    img = read_img(img)
    h, w = img.shape[:2]
    new_size = scale_size((w, h), ratio)
    return cv2.resize(img, new_size, interpolation=interpolation)


def resize_keep_ar(img,
                   max_long_edge,
                   max_short_edge,
                   return_scale=False,
                   interpolation=INTER_LINEAR):
    """Resize image with aspect ratio unchanged
    the long edge of resized image is no greater than max_long_edge, the short
    edge of resized image is no greater than max_short_edge.

    Args:
        img(ndarray): image or image path
        max_long_edge(int): max value of the long edge of resized image
        max_short_edge(int): max value of the short edge of resized image
        return_scale(bool): whether to return scale besides the resized image
        interpolation(enum): interpolation method

    Returns:
        tuple: (resized image, scale factor)
    """
    if max_long_edge < max_short_edge:
        raise ValueError(
            '"max_long_edge" should not be less than "max_short_edge"')
    img = read_img(img)
    h, w = img.shape[:2]
    scale = min(
        float(max_long_edge) / max(h, w), float(max_short_edge) / min(h, w))
    new_size = scale_size((w, h), scale)
    resized_img = cv2.resize(img, new_size, interpolation=interpolation)
    if return_scale:
        return resized_img, scale
    else:
        return resized_img


def limit_size(img, max_edge, return_scale=False, interpolation=INTER_LINEAR):
    """Limit the size of an image
    If the long edge of the image is greater than max_edge, resize the image

    Args:
        img(ndarray): input image
        max_edge(int): max value of long edge
        return_scale(bool): whether to return scale besides the resized image
        interpolation(enum): interpolation method

    Returns:
        tuple: (resized image, scale factor)
    """
    img = read_img(img)
    h, w = img.shape[:2]
    if max(h, w) > max_edge:
        scale = float(max_edge) / max(h, w)
        new_size = scale_size((w, h), scale)
        resized_img = cv2.resize(img, new_size, interpolation=interpolation)
    else:
        scale = 1.0
        resized_img = img
    if return_scale:
        return resized_img, scale
    else:
        return resized_img
