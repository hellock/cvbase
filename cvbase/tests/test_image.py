from os import path

import pytest
from cvbase.image import read_img, img_from_bytes, resize_keep_ar, limit_size


class TestImage(object):

    @classmethod
    def setup_class(cls):
        cls.img_path = path.join(path.dirname(__file__), 'data/test.jpg')

    def test_read_img(self):
        img = read_img(self.img_path)
        assert img.shape == (300, 400, 3)

    def test_img_from_bytes(self):
        img_bytes = open(self.img_path, 'rb').read()
        img = img_from_bytes(img_bytes)
        assert img.shape == (300, 400, 3)

    def test_resize_keep_ar(self):
        # resize (400, 300) to (max_1000, max_600)
        resized_img = resize_keep_ar(self.img_path, 1000, 600)
        assert resized_img.shape == (600, 800, 3)
        resized_img, scale = resize_keep_ar(self.img_path, 1000, 600, True)
        assert resized_img.shape == (600, 800, 3) and scale == 2.0
        # resize (400, 300) to (max_200, max_180)
        img = read_img(self.img_path)
        resized_img = resize_keep_ar(img, 200, 180)
        assert resized_img.shape == (150, 200, 3)
        resized_img, scale = resize_keep_ar(self.img_path, 200, 180, True)
        assert resized_img.shape == (150, 200, 3) and scale == 0.5
        # max_long_edge cannot be less than max_short_edge
        with pytest.raises(ValueError):
            resize_keep_ar(self.img_path, 500, 600)

    def test_limit_size(self):
        # limit to 800
        resized_img = limit_size(self.img_path, 800)
        assert resized_img.shape == (300, 400, 3)
        resized_img, scale = limit_size(self.img_path, 800, True)
        assert resized_img.shape == (300, 400, 3) and scale == 1
        # limit to 200
        resized_img = limit_size(self.img_path, 200)
        assert resized_img.shape == (150, 200, 3)
        resized_img, scale = limit_size(self.img_path, 200, True)
        assert resized_img.shape == (150, 200, 3) and scale == 0.5
        # test with img rather than img path
        img = read_img(self.img_path)
        resized_img = limit_size(img, 200)
        assert resized_img.shape == (150, 200, 3)
        resized_img, scale = limit_size(img, 200, True)
        assert resized_img.shape == (150, 200, 3) and scale == 0.5
