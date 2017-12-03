from io import BytesIO
from os import path

import numpy as np
import PIL.Image as PImage
import PIL.ImageDraw as PImageDraw
import PIL.ImageFont as PImageFont

from cvbase.det import bbox_clip, bbox_scaling
from cvbase.image import ImageBase, ImageDrawBase


class Image(ImageBase):

    @classmethod
    def open(cls, filename):
        return Image(PImage.open(filename))

    @classmethod
    def new(cls, mode, size, color=0):
        return Image(PImage.new(mode, size, color))

    @classmethod
    def frombytes(cls, content, mode=None, size=None, **kwargs):
        if size is None:
            return Image(PImage.open(BytesIO(content)))
        else:
            return Image(PImage.frombytes(mode, size, content, **kwargs))

    @classmethod
    def fromarray(cls, array, mode=None):
        return Image(PImage.fromarray(array, mode))

    @classmethod
    def scale_size(cls, size, ratio):
        w, h = size
        return int(w * float(ratio) + 0.5), int(h * float(ratio) + 0.5)

    def __init__(self, img_data, mode='RGB', filename=None):
        assert isinstance(img_data, PImage.Image)
        self.data = img_data

    def __copy__(self):
        return self.__class__(self.data.copy())

    @property
    def filename(self):
        return self.data.filename

    @property
    def mode(self):
        return self.data.mode

    @property
    def size(self):
        return self.data.size

    @property
    def channels(self):
        return len(self.data.getbands())

    def copy(self):
        return self.__copy__()

    def convert(self, mode):
        return Image(self.data.convert(mode))

    def resize(self, size, return_scale=False, resample='bilinear', box=None):
        resample_methods = {
            'nearest': PImage.NEAREST,
            'bilinear': PImage.BILINEAR,
            'hamming': PImage.HAMMING,
            'bicubic': PImage.BICUBIC,
            'area': PImage.BOX,
            'lanczos': PImage.LANCZOS
        }
        if resample not in resample_methods:
            raise ValueError(
                '{} is not a valid interpolation type'.format(resample))
        resized_img = self.data.resize(
            size, resample=resample_methods[resample], box=box)
        resized_img = Image(resized_img)
        if return_scale:
            if box is None:
                box_size = self.size
            else:
                box_size = (box[2] - box[0] + 1, box[3] - box[1] + 1)
            w_ratio = float(size[0]) / box_size[0]
            h_ratio = float(size[1]) / box_size[1]
            return resized_img, w_ratio, h_ratio
        else:
            return resized_img

    def flip(self, method='horizontal'):
        flip_method = {
            'horizontal': PImage.FLIP_LEFT_RIGHT,
            'vertical': PImage.FLIP_TOP_BOTTOM
        }
        if method not in flip_method:
            raise ValueError('Invalid flip method {}'.format(method))
        return Image(self.data.transpose(flip_method[method]))

    def crop(self, bboxes, scale_ratio=1.0, pad_fill=None):
        if pad_fill is not None:
            if isinstance(pad_fill, (int, float)):
                pad_fill = tuple([pad_fill for _ in range(self.channels)])
            assert len(pad_fill) == self.channels
        _bboxes = bboxes[np.newaxis, ...] if bboxes.ndim == 1 else bboxes
        scaled_bboxes = bbox_scaling(_bboxes, scale_ratio)
        scaled_bboxes = scaled_bboxes.astype(np.int32)
        clipped_bbox = bbox_clip(scaled_bboxes, self.size)
        patches = []
        for i in range(clipped_bbox.shape[0]):
            x1, y1, x2, y2 = tuple(clipped_bbox[i, :].tolist())
            if pad_fill is None:
                patch = self.data.crop((x1, y1, x2 + 1, y2 + 1))
            else:
                _x1, _y1, _x2, _y2 = tuple(scaled_bboxes[i, :].tolist())
                patch_size = (_x2 - _x1 + 1, _y2 - _y1 + 1)
                patch = PImage.new(self.mode, patch_size, pad_fill)
                x_start = 0 if _x1 >= 0 else -_x1
                y_start = 0 if _y1 >= 0 else -_y1
                region = self.data.crop((x1, y1, x2 + 1, y2 + 1))
                patch.paste(region, (x_start, y_start))
            patches.append(Image(patch))
        if bboxes.ndim == 1:
            return patches[0]
        else:
            return patches

    def pad(self, size, pad_val):
        if not isinstance(pad_val, (int, float)):
            assert len(pad_val) == self.channels
        assert size[0] >= self.size[0] and size[1] >= self.size[1]
        pad = PImage.new(self.mode, size, pad_val)
        pad.paste(self.data)
        return Image(pad)

    def save(self, filename, **params):
        self.data.save(filename, **params)

    def show(self, title=None, command=None):
        self.data.show(title, command)

    def numpy(self, dtype=None):
        if dtype is None:
            return np.array(self.data)
        else:
            return np.array(self.data, dtype=dtype)


class ImageDraw(ImageDrawBase):

    font_path = path.join(path.dirname(__file__), 'fonts/opensans.ttf')
    font = PImageFont.truetype(font_path, 12)

    @classmethod
    def load_font(cls, font_path, font_size=12):
        cls.font_path = font_path
        cls.font = PImageFont.truetype(font_path, font_size)

    def __init__(self, img, mode=None):
        super(ImageDraw, self).__init__(img, mode)
        self.img_draw = PImageDraw.Draw(self.img.data)

    def line(self, start, end, color=None, thickness=1):
        self.img_draw.line([start, end], color, thickness)

    def rectangle(self, bbox, color=None, thickness=1):
        bbox = bbox.tolist() if isinstance(bbox, np.ndarray) else bbox
        fill = None if thickness >= 0 else color
        self.img_draw.rectangle(bbox, fill=fill, outline=color)

    def ellipse(self, bbox, color=None, thickness=1):
        bbox = bbox.tolist() if isinstance(bbox, np.ndarray) else bbox
        fill = None if thickness >= 0 else color
        self.img_draw.ellipse(bbox, fill=fill, outline=color)

    def text(self, text, xy, color=None, font=None, font_size=None, **kwargs):
        if font is None:
            if font_size and font_size != self.font.size:
                self.load_font(self.font_path, font_size)
        else:
            self.font = font
        if isinstance(xy, np.ndarray):
            xy = xy.tolist()
        text_height = self.font.getsize(' ')[1]
        self.img_draw.text((xy[0], xy[1] - text_height), text, color,
                           self.font)
