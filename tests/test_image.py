import os
from os import path

import cvbase as cvb
import numpy as np
import pytest
from numpy.testing import assert_array_equal, assert_array_almost_equal


class TestImage(object):

    @classmethod
    def setup_class(cls):
        # the test img resolution is 400x300
        cls.img_path = path.join(path.dirname(__file__), 'data/color.jpg')
        cls.gray_img_path = path.join(
            path.dirname(__file__), 'data/grayscale.jpg')

    def test_read_img(self):
        img = cvb.read_img(self.img_path)
        assert img.shape == (300, 400, 3)
        img = cvb.read_img(self.img_path, cvb.IMREAD_GRAYSCALE)
        assert img.shape == (300, 400)
        img = cvb.read_img(self.gray_img_path)
        assert img.shape == (300, 400, 3)
        img = cvb.read_img(self.gray_img_path, cvb.IMREAD_UNCHANGED)
        assert img.shape == (300, 400)
        img = cvb.read_img(img)
        assert_array_equal(img, cvb.read_img(img))
        with pytest.raises(TypeError):
            cvb.read_img(1)

    def test_img_from_bytes(self):
        with open(self.img_path, 'rb') as f:
            img_bytes = f.read()
        img = cvb.img_from_bytes(img_bytes)
        assert img.shape == (300, 400, 3)

    def test_write_img(self):
        img = cvb.read_img(self.img_path)
        out_file = '.cvbase_test.tmp.jpg'
        cvb.write_img(img, out_file)
        assert cvb.read_img(out_file).shape == (300, 400, 3)
        os.remove(out_file)

    def test_bgr2gray(self):
        in_img = np.random.rand(10, 10, 3).astype(np.float32)
        out_img = cvb.bgr2gray(in_img)
        computed_gray = (in_img[:, :, 0] * 0.114 + in_img[:, :, 1] * 0.587 +
                         in_img[:, :, 2] * 0.299)
        assert_array_almost_equal(out_img, computed_gray, decimal=4)
        out_img_3d = cvb.bgr2gray(in_img, True)
        assert out_img_3d.shape == (10, 10, 1)
        assert_array_almost_equal(out_img_3d[..., 0], out_img, decimal=4)

    def test_gray2bgr(self):
        in_img = np.random.rand(10, 10).astype(np.float32)
        out_img = cvb.gray2bgr(in_img)
        assert out_img.shape == (10, 10, 3)
        for i in range(3):
            assert_array_almost_equal(out_img[..., i], in_img, decimal=4)

    def test_bgr2rgb(self):
        in_img = np.random.rand(10, 10, 3).astype(np.float32)
        out_img = cvb.bgr2rgb(in_img)
        assert out_img.shape == in_img.shape
        assert_array_equal(out_img[..., 0], in_img[..., 2])
        assert_array_equal(out_img[..., 1], in_img[..., 1])
        assert_array_equal(out_img[..., 2], in_img[..., 0])

    def test_rgb2bgr(self):
        in_img = np.random.rand(10, 10, 3).astype(np.float32)
        out_img = cvb.rgb2bgr(in_img)
        assert out_img.shape == in_img.shape
        assert_array_equal(out_img[..., 0], in_img[..., 2])
        assert_array_equal(out_img[..., 1], in_img[..., 1])
        assert_array_equal(out_img[..., 2], in_img[..., 0])

    def test_bgr2hsv(self):
        in_img = np.random.rand(10, 10, 3).astype(np.float32)
        out_img = cvb.bgr2hsv(in_img)
        argmax = in_img.argmax(axis=2)
        computed_hsv = np.empty_like(in_img, dtype=in_img.dtype)
        for i in range(in_img.shape[0]):
            for j in range(in_img.shape[1]):
                b = in_img[i, j, 0]
                g = in_img[i, j, 1]
                r = in_img[i, j, 2]
                v = max(r, g, b)
                s = (v - min(r, g, b)) / v if v != 0 else 0
                if argmax[i, j] == 0:
                    h = 240 + 60 * (r - g) / (v - min(r, g, b))
                elif argmax[i, j] == 1:
                    h = 120 + 60 * (b - r) / (v - min(r, g, b))
                else:
                    h = 60 * (g - b) / (v - min(r, g, b))
                if h < 0:
                    h += 360
                computed_hsv[i, j, :] = [h, s, v]
        assert_array_almost_equal(out_img, computed_hsv, decimal=3)

    def test_scale_size(self):
        assert cvb.scale_size((300, 200), 0.5) == (150, 100)
        assert cvb.scale_size((11, 22), 0.7) == (8, 15)

    def test_resize(self):
        resized_img = cvb.resize(self.img_path, (1000, 600))
        assert resized_img.shape == (600, 1000, 3)
        resized_img, w_scale, h_scale = cvb.resize(self.img_path, (1000, 600),
                                                   True)
        assert (resized_img.shape == (600, 1000, 3) and w_scale == 2.5 and
                h_scale == 2.0)
        for mode in [
                cvb.INTER_NEAREST, cvb.INTER_LINEAR, cvb.INTER_CUBIC,
                cvb.INTER_AREA, cvb.INTER_LANCZOS4
        ]:
            resized_img = cvb.resize(
                self.img_path, (1000, 600), interpolation=mode)
            assert resized_img.shape == (600, 1000, 3)

    def test_resize_like(self):
        a = np.zeros((100, 200, 3))
        resized_img = cvb.resize_like(self.img_path, a)
        assert resized_img.shape == (100, 200, 3)

    def test_resize_by_ratio(self):
        resized_img = cvb.resize_by_ratio(self.img_path, 1.5)
        assert resized_img.shape == (450, 600, 3)
        resized_img = cvb.resize_by_ratio(self.img_path, 0.934)
        assert resized_img.shape == (280, 374, 3)

    def test_resize_keep_ar(self):
        # resize (400, 300) to (max_1000, max_600)
        resized_img = cvb.resize_keep_ar(self.img_path, 1000, 600)
        assert resized_img.shape == (600, 800, 3)
        resized_img, scale = cvb.resize_keep_ar(self.img_path, 1000, 600, True)
        assert resized_img.shape == (600, 800, 3) and scale == 2.0
        # resize (400, 300) to (max_200, max_180)
        img = cvb.read_img(self.img_path)
        resized_img = cvb.resize_keep_ar(img, 200, 180)
        assert resized_img.shape == (150, 200, 3)
        resized_img, scale = cvb.resize_keep_ar(self.img_path, 200, 180, True)
        assert resized_img.shape == (150, 200, 3) and scale == 0.5
        # max_long_edge cannot be less than max_short_edge
        with pytest.raises(ValueError):
            cvb.resize_keep_ar(self.img_path, 500, 600)

    def test_limit_size(self):
        # limit to 800
        resized_img = cvb.limit_size(self.img_path, 800)
        assert resized_img.shape == (300, 400, 3)
        resized_img, scale = cvb.limit_size(self.img_path, 800, True)
        assert resized_img.shape == (300, 400, 3) and scale == 1
        # limit to 200
        resized_img = cvb.limit_size(self.img_path, 200)
        assert resized_img.shape == (150, 200, 3)
        resized_img, scale = cvb.limit_size(self.img_path, 200, True)
        assert resized_img.shape == (150, 200, 3) and scale == 0.5
        # test with img rather than img path
        img = cvb.read_img(self.img_path)
        resized_img = cvb.limit_size(img, 200)
        assert resized_img.shape == (150, 200, 3)
        resized_img, scale = cvb.limit_size(img, 200, True)
        assert resized_img.shape == (150, 200, 3) and scale == 0.5

    def test_crop_img(self):
        img = cvb.read_img(self.img_path)
        # yapf: disable
        bboxes = np.array([[100, 100, 199, 199],  # center
                           [0, 0, 150, 100],  # left-top corner
                           [250, 200, 399, 299],  # right-bottom corner
                           [0, 100, 399, 199],  # wide
                           [150, 0, 299, 299]])  # tall
        # yapf: enable
        # crop one bbox
        patch = cvb.crop_img(img, bboxes[0, :])
        patches = cvb.crop_img(img, bboxes[[0], :])
        assert patch.shape == (100, 100, 3)
        patch_path = path.join(path.dirname(__file__), 'data/patches')
        ref_patch = np.load(patch_path + '/0.npy')
        assert_array_equal(patch, ref_patch)
        assert isinstance(patches, list) and len(patches) == 1
        assert_array_equal(patches[0], ref_patch)
        # crop with no scaling and padding
        patches = cvb.crop_img(img, bboxes)
        assert len(patches) == bboxes.shape[0]
        for i in range(len(patches)):
            ref_patch = np.load(patch_path + '/{}.npy'.format(i))
            assert_array_equal(patches[i], ref_patch)
        # crop with scaling and no padding
        patches = cvb.crop_img(img, bboxes, 1.2)
        for i in range(len(patches)):
            ref_patch = np.load(patch_path + '/scale_{}.npy'.format(i))
            assert_array_equal(patches[i], ref_patch)
        # crop with scaling and padding
        patches = cvb.crop_img(img, bboxes, 1.2, pad_fill=[255, 255, 0])
        for i in range(len(patches)):
            ref_patch = np.load(patch_path + '/pad_{}.npy'.format(i))
            assert_array_equal(patches[i], ref_patch)
        patches = cvb.crop_img(img, bboxes, 1.2, pad_fill=0)
        for i in range(len(patches)):
            ref_patch = np.load(patch_path + '/pad0_{}.npy'.format(i))
            assert_array_equal(patches[i], ref_patch)
