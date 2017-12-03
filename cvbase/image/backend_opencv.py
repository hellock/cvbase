import cv2
import numpy as np

from cvbase.det import bbox_clip, bbox_scaling
from cvbase.image import ImageBase, ImageDrawBase
from cvbase.io import check_file_exist
from cvbase.opencv import IMREAD_UNCHANGED


class Image(ImageBase):

    @classmethod
    def frombytes(cls, content, mode='BGR'):
        img_np = np.fromstring(content, np.uint8)
        img = cv2.imdecode(img_np, flags=-1)
        return Image(img, mode=mode)

    @classmethod
    def fromarray(cls, array, mode=None):
        assert isinstance(array, np.ndarray)
        if mode is None:
            if len(array.shape) == 3 and array.shape[-1] == 3:
                mode = 'BGR'
            elif len(array.shape) == 4:
                mode = 'BGRA'
            elif len(array.shape) == 2:
                mode = 'L'
        return Image(np.ascontiguousarray(array), mode=mode)

    @classmethod
    def open(cls, filename):
        check_file_exist(filename,
                         'img file does not exist: {}'.format(filename))
        data = cv2.imread(filename, IMREAD_UNCHANGED)
        img = Image(data, filename=filename)
        return img

    @classmethod
    def new(cls, mode, size, color=0):
        shape = (size[1], size[0], 3) if mode != 'L' else (size[1], size[0])
        channels = shape[-1] if len(shape) > 2 else 1
        if isinstance(color, int):
            dtype = 'uint8'
        elif isinstance(color, float):
            dtype = 'float32'
        elif isinstance(color, (tuple, list)):
            assert len(color) == channels
            dtype = 'uint8'
            for val in color:
                if isinstance(val, float):
                    dtype = 'float32'
                    break
        else:
            raise TypeError('color must be a number of tuple')
        data = np.empty(shape, dtype)
        if channels == 1 or isinstance(color, (int, float)):
            data.fill(color)
        else:
            for i in range(channels):
                data[..., i] = color[i]
        img = Image(data, mode=mode)
        return img

    @classmethod
    def scale_size(cls, size, ratio):
        w, h = size
        return int(w * float(ratio) + 0.5), int(h * float(ratio) + 0.5)

    def __init__(self, img_data, mode='BGR', filename=None):
        assert isinstance(img_data, np.ndarray)
        super(Image, self).__init__(img_data, mode, filename)

    def __copy__(self):
        return self.__class__(self.data.copy(), self.filename, self.mode)

    @property
    def size(self):
        return self.data.shape[1::-1]

    @property
    def channels(self):
        return self.data.shape[-1] if len(self.data.shape) > 2 else 1

    def copy(self):
        return self.__copy__()

    def convert(self, mode):
        if mode == self.mode:
            return self.copy()
        color_codes = {
            'RGB': {
                'L': cv2.COLOR_RGB2GRAY,
                'BGR': cv2.COLOR_RGB2BGR,
                'HSV': cv2.COLOR_RGB2HSV
            },
            'BGR': {
                'L': cv2.COLOR_BGR2GRAY,
                'RGB': cv2.COLOR_BGR2RGB,
                'HSV': cv2.COLOR_BGR2HSV
            },
            'L': {
                'BGR': cv2.COLOR_GRAY2BGR,
                'RGB': cv2.COLOR_GRAY2RGB
            },
            'HSV': {
                'BGR': cv2.COLOR_HSV2BGR,
                'RGB': cv2.COLOR_HSV2RGB
            }
        }
        try:
            new_img = cv2.cvtColor(self.data, color_codes[self.mode][mode])
        except KeyError:
            raise ValueError('converting from {} to {} is not supported'.
                             format(self.mode, mode))
        return Image(new_img, mode=mode)

    def resize(self, size, return_scale=False, resample='bilinear', box=None):
        resample_methods = {
            'nearest': cv2.INTER_NEAREST,
            'bilinear': cv2.INTER_LINEAR,
            'bicubic': cv2.INTER_CUBIC,
            'area': cv2.INTER_AREA,
            'lanczos': cv2.INTER_LANCZOS4
        }
        if resample not in resample_methods:
            raise ValueError(
                '{} is not a valid interpolation type'.format(resample))
        region = self.data[box[1]:box[3], box[0]:box[2]] if box else self.data
        resized_img = cv2.resize(
            region, size, interpolation=resample_methods[resample])
        resized_img = Image(resized_img, self.filename, self.mode)
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
        flip_method = {'horizontal': 1, 'vertical': 0}
        if method not in flip_method:
            raise ValueError('Invalid flip method {}'.format(method))
        flipped = cv2.flip(self.data, flip_method[method])
        return Image(flipped, self.mode, self.filename)

    def crop(self, bboxes, scale_ratio=1.0, pad_fill=None):
        if pad_fill is not None:
            if isinstance(pad_fill, (int, float)):
                pad_fill = [pad_fill for _ in range(self.channels)]
            assert len(pad_fill) == self.channels
        _bboxes = bboxes[np.newaxis, ...] if bboxes.ndim == 1 else bboxes
        scaled_bboxes = bbox_scaling(_bboxes, scale_ratio)
        scaled_bboxes = scaled_bboxes.astype(np.int32)
        clipped_bbox = bbox_clip(scaled_bboxes, self.size)
        patches = []
        for i in range(clipped_bbox.shape[0]):
            x1, y1, x2, y2 = tuple(clipped_bbox[i, :].tolist())
            if pad_fill is None:
                patch = self.data[y1:y2 + 1, x1:x2 + 1, ...]
                print(x1, y1, x2, y2, patch.shape)
            else:
                _x1, _y1, _x2, _y2 = tuple(scaled_bboxes[i, :].tolist())
                if self.channels == 1:
                    patch_shape = (_y2 - _y1 + 1, _x2 - _x1 + 1)
                else:
                    patch_shape = (_y2 - _y1 + 1, _x2 - _x1 + 1, self.channels)
                patch = np.array(
                    pad_fill, dtype=self.data.dtype) * np.ones(
                        patch_shape, dtype=self.data.dtype)
                x_start = 0 if _x1 >= 0 else -_x1
                y_start = 0 if _y1 >= 0 else -_y1
                w = x2 - x1 + 1
                h = y2 - y1 + 1
                patch[y_start:y_start + h, x_start:x_start + w, ...] = (
                    self.data[y1:y1 + h, x1:x1 + w, ...])
            patches.append(Image(patch, self.mode, self.filename))
        if bboxes.ndim == 1:
            return patches[0]
        else:
            return patches

    def pad(self, size, pad_val):
        if not isinstance(pad_val, (int, float)):
            assert len(pad_val) == self.channels
        assert size[0] >= self.size[0] and size[1] >= self.size[1]
        shape = ((size[1], size[0], self.data.shape[-1])
                 if len(size) < len(self.data.shape) else size)
        assert len(shape) == len(self.data.shape)
        for i in range(len(shape) - 1):
            assert shape[i] >= self.data.shape[i]
        pad = np.empty(shape, dtype=self.data.dtype)
        pad[...] = pad_val
        pad[:self.height, :self.width, ...] = self.data
        return Image(pad, self.filename, self.mode)

    def save(self, filename, params=None):
        cv2.imwrite(filename, self.data, params)

    def show(self, title=None, wait_time=0):
        cv2.imshow(title, self.data)
        if wait_time >= 0:
            cv2.waitKey(wait_time)

    def numpy(self, dtype=None):
        if dtype is None:
            return self.data
        else:
            return self.data.astype(dtype)


class ImageDraw(ImageDrawBase):

    def line(self, start, end, color=None, thickness=1):
        cv2.line(self.img.data, start, end, color, thickness)

    def rectangle(self, bbox, color=None, thickness=1):
        cv2.rectangle(self.img.data, (bbox[0], bbox[1]), (bbox[2], bbox[3]),
                      color, thickness)

    def ellipse(self, bbox, color=None, thickness=1):
        cv2.ellipse(self.img.data, bbox, color, thickness)

    def text(self,
             text,
             position,
             color=None,
             font=None,
             font_size=12,
             thickness=1):
        color = (0, 0, 0) if color is None else color
        font = cv2.FONT_HERSHEY_COMPLEX if font is None else font
        font_scale = font_size / 21.0
        cv2.putText(self.img.data, text, position, font, font_scale, color,
                    thickness)
