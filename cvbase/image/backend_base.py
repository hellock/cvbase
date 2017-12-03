import numpy as np


class ImageBase(object):

    @classmethod
    def open(cls, filename):
        raise NotImplementedError

    @classmethod
    def new(cls, mode, size, color=0):
        raise NotImplementedError

    @classmethod
    def frombytes(cls, content, mode='BGR'):
        raise NotImplementedError

    @classmethod
    def fromarray(cls, array, mode=None):
        raise NotImplementedError

    @classmethod
    def scale_size(cls, size, ratio):
        w, h = size
        return int(w * float(ratio) + 0.5), int(h * float(ratio) + 0.5)

    def __init__(self, img_data, mode='BGR', filename=None):
        self.data = img_data
        self._mode = mode
        self._filename = filename
        if self.channels == 1:
            self._mode = 'L'

    def __copy__(self):
        return self.__class__(self.data.copy(), self.filename, self.mode)

    @property
    def filename(self):
        return self._filename

    @property
    def mode(self):
        return self._mode

    @property
    def size(self):
        raise NotImplementedError

    @property
    def width(self):
        return self.size[0]

    @property
    def height(self):
        return self.size[1]

    @property
    def long_edge(self):
        return max(self.size)

    @property
    def short_edge(self):
        return min(self.size)

    @property
    def channels(self):
        raise NotImplementedError

    def copy(self):
        return self.__copy__()

    def convert(self, mode):
        raise NotImplementedError

    def resize(self, size, return_scale=False, resample='bilinear', box=None):
        raise NotImplementedError

    def resize_like(self, dst_img, return_scale=False, resample='bilinear'):
        if isinstance(dst_img, ImageBase):
            return self.resize(dst_img.size, return_scale, resample)
        elif isinstance(dst_img, np.ndarray):
            return self.resize(dst_img.shape[1::-1], return_scale, resample)
        else:
            raise TypeError('"dst_img" must be an Image or a numpy array')

    def resize_by_ratio(self, ratio, resample='bilinear', box=None):
        return self.resize(
            self.scale_size(self.size, ratio), resample=resample, box=box)

    def resize_keep_ar(self,
                       max_long_edge,
                       max_short_edge,
                       return_scale=False,
                       resample='bilinear'):
        if max_long_edge < max_short_edge:
            raise ValueError(
                '"max_long_edge" should not be less than "max_short_edge"')
        ratio = min(
            float(max_long_edge) / self.long_edge,
            float(max_short_edge) / self.short_edge)
        new_size = self.scale_size(self.size, ratio)
        resized_img = self.resize(new_size, resample=resample)
        if return_scale:
            return resized_img, ratio
        else:
            return resized_img

    def limit_size(self, max_edge, return_scale=False, resample='bilinear'):
        if self.long_edge > max_edge:
            ratio = float(max_edge) / self.long_edge
            new_size = self.scale_size(self.size, ratio)
            resized_img = self.resize(new_size, resample=resample)
        else:
            ratio = 1.0
            resized_img = self.copy()
        if return_scale:
            return resized_img, ratio
        else:
            return resized_img

    def flip(self, method='horizontal'):
        raise NotImplementedError

    def crop(self, bboxes, scale_ratio=1.0, pad_fill=None):
        raise NotImplementedError

    def pad(self, size, pad_val):
        raise NotImplementedError

    def save(self, filename, **params):
        raise NotImplementedError

    def show(self, title=None, **kwargs):
        raise NotImplementedError

    def numpy(self, dtype=None):
        raise NotImplementedError


class ImageDrawBase(object):

    def __init__(self, img, mode=None):
        self.img = img
        if mode:
            self._mode = mode
        else:
            self._mode = img.mode

    @property
    def mode(self):
        return self._mode

    def line(self, start, end, color=None, thickness=1):
        raise NotImplementedError

    def rectangle(self, bbox, color=None, thickness=1):
        raise NotImplementedError

    def ellipse(self, bbox, color=None, thickness=1):
        raise NotImplementedError

    def text(self,
             text,
             position,
             color=None,
             font=None,
             font_size=None,
             **kwargs):
        raise NotImplementedError
