from os import path

import pytest
from cvbase import (read_img, img_from_bytes, resize, resize_by_ratio,
                    resize_keep_ar, limit_size)


class TestImage(object):

    @classmethod
    def setup_class(cls):
        # the test img shape is (300, 400, 3)
        cls.img_path = path.join(path.dirname(__file__), 'data/test.jpg')

    def test_read_img(self):
        img = read_img(self.img_path)
        assert img.shape == (300, 400, 3)

    def test_img_from_bytes(self):
        img_bytes = open(self.img_path, 'rb').read()
        img = img_from_bytes(img_bytes)
        assert img.shape == (300, 400, 3)

    def test_resize(self):
        resized_img = resize(self.img_path, (1000, 600))
        assert resized_img.shape == (600, 1000, 3)
        resized_img, w_scale, h_scale = resize(self.img_path, (1000, 600),
                                               True)
        assert (resized_img.shape == (600, 1000, 3) and w_scale == 2.5 and
                h_scale == 2.0)

    def test_resize_by_ratio(self):
        resized_img = resize_by_ratio(self.img_path, 1.5)
        assert resized_img.shape == (450, 600, 3)
        resized_img = resize_by_ratio(self.img_path, 0.934)
        assert resized_img.shape == (280, 374, 3)

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
