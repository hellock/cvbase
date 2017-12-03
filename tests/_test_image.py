import sys
import tempfile
from os import path

import cvbase as cvb
import cvbase.image as im
import numpy as np
import pytest
from numpy.testing import assert_array_equal, assert_array_almost_equal


class _TestImage(object):

    @classmethod
    def setup_class(cls):
        # the test img resolution is 400x300
        cls.img_path = path.join(path.dirname(__file__), 'data/color.jpg')
        cls.gray_img_path = path.join(
            path.dirname(__file__), 'data/grayscale.jpg')

    def assert_img_equal(self, img, ref_img, ratio_thr=0.99):
        assert img.shape == ref_img.shape
        assert img.dtype == ref_img.dtype
        area = ref_img.shape[0] * ref_img.shape[1]
        diff = np.abs(img.astype('int32') - ref_img.astype('int32'))
        assert np.sum(diff <= 1) / float(area) > ratio_thr

    def test_backend(self):
        assert im.Image is getattr(im, 'backend_' + cvb.get_backend()).Image

    def test_open(self):
        img = im.Image.open(self.img_path)
        assert img.filename == self.img_path
        if cvb.get_backend() == 'opencv':
            assert img.mode == 'BGR'
        elif cvb.get_backend() == 'pillow':
            assert img.mode == 'RGB'
        assert img.channels == 3
        assert img.size == (400, 300)
        assert img.width == 400
        assert img.height == 300
        assert img.long_edge == 400
        assert img.short_edge == 300
        img = im.Image.open(self.gray_img_path)
        assert img.filename == self.gray_img_path
        assert img.mode == 'L'
        assert img.channels == 1
        assert img.size == (400, 300)
        assert img.width == 400
        assert img.height == 300
        assert img.long_edge == 400
        assert img.short_edge == 300
        if sys.version_info > (3, 3):
            with pytest.raises(FileNotFoundError):
                im.Image.open('a.jpg')
        else:
            with pytest.raises(IOError):
                im.Image.open('a.jpg')

    def test_new(self):
        img = im.Image.new('RGB', (200, 100))
        assert (img.numpy().shape == (100, 200, 3)
                and img.numpy().dtype == np.uint8)
        img = im.Image.new('RGB', (200, 100), (0, 100, 50))
        img_np = img.numpy()
        assert img_np.shape == (100, 200, 3)
        assert img_np.dtype == np.uint8
        assert (np.all(img_np[:, :, 0] == 0)
                and np.all(img_np[:, :, 1] == 100)
                and np.all(img_np[:, :, 2] == 50))
        img = im.Image.new('L', (200, 100), 50)
        img_np = img.numpy()
        assert (img_np.shape == (100, 200) and img_np.dtype == np.uint8
                and np.all(img_np == 50))

    def test_frombytes(self):
        with open(self.img_path, 'rb') as f:
            img_bytes = f.read()
        img = im.Image.frombytes(img_bytes)
        assert img.numpy().shape == (300, 400, 3)

    def test_fromarray(self):
        data = np.random.randint(256, size=(10, 20, 3), dtype='uint8')
        img = im.Image.fromarray(data)
        mode = 'BGR' if cvb.get_backend() == 'opencv' else 'RGB'
        assert img.size == (20, 10) and img.mode == mode
        assert_array_equal(img.numpy(), data)
        data = np.random.randint(256, size=(10, 20, 3), dtype='uint8')
        img = im.Image.fromarray(data, 'HSV')
        assert img.size == (20, 10) and img.mode == 'HSV'
        assert_array_equal(img.numpy(), data)
        data = np.random.randint(256, size=(10, 20), dtype='uint8')
        img = im.Image.fromarray(data)
        assert img.size == (20, 10) and img.mode == 'L'
        assert_array_equal(img.numpy(), data)
        # with pytest.raises(AssertionError):
        #     im.Image.fromarray([[1, 2, 3], [1, 2, 3]])

    def test_save(self):
        img = im.Image.open(self.img_path)
        out_file = path.join(tempfile.mkdtemp(), 'test_save.jpg')
        img.save(out_file)
        assert im.Image.open(out_file).numpy().shape == (300, 400, 3)

    def test_rgb2gray(self):
        in_img = np.random.randint(256, size=(100, 100, 3), dtype='uint8')
        out_img = im.Image.fromarray(in_img, mode='RGB').convert('L').numpy()
        computed_gray = (in_img[:, :, 0] * 0.299 + in_img[:, :, 1] * 0.587 +
                         in_img[:, :, 2] * 0.114).astype(np.uint8)
        self.assert_img_equal(out_img, computed_gray)

    def test_gray2rgb(self):
        in_img = np.random.randint(256, size=(100, 100), dtype='uint8')
        out_img = im.Image.fromarray(in_img).convert('RGB').numpy()
        assert out_img.shape == (100, 100, 3)
        for i in range(3):
            assert_array_almost_equal(out_img[..., i], in_img, decimal=4)

    def test_bgr2rgb(self):
        in_img = np.random.randint(256, size=(100, 100, 3), dtype='uint8')
        out_img = im.Image.fromarray(in_img).convert('RGB').numpy()
        assert out_img.shape == in_img.shape
        assert_array_equal(out_img[..., 0], in_img[..., 2])
        assert_array_equal(out_img[..., 1], in_img[..., 1])
        assert_array_equal(out_img[..., 2], in_img[..., 0])

    def test_rgb2bgr(self):
        in_img = np.random.randint(256, size=(100, 100, 3), dtype='uint8')
        out_img = im.Image.fromarray(in_img, 'RGB').convert('BGR').numpy()
        assert out_img.shape == in_img.shape
        assert_array_equal(out_img[..., 0], in_img[..., 2])
        assert_array_equal(out_img[..., 1], in_img[..., 1])
        assert_array_equal(out_img[..., 2], in_img[..., 0])

    def test_rgb2hsv(self):
        in_img = np.random.randint(256, size=(100, 100, 3), dtype='uint8')
        out_img = im.Image.fromarray(in_img, mode='RGB').convert('HSV').numpy()
        argmax = in_img.argmax(axis=2)
        computed_hsv = np.empty_like(in_img, dtype=np.float32)
        for i in range(in_img.shape[0]):
            for j in range(in_img.shape[1]):
                r, g, b = tuple(
                    (in_img[i, j, :].astype(np.float32) / 255).tolist())
                v = max(r, g, b)
                u = min(r, g, b)
                s = (v - u) / v if v != 0 else 0
                if argmax[i, j] == 0:  # v = r
                    h = 60 * (g - b) / (v - u + 1e-4)
                elif argmax[i, j] == 1:  # v = g
                    h = 120 + 60 * (b - r) / (v - u + 1e-4)
                else:  # v = b
                    h = 240 + 60 * (r - g) / (v - u + 1e-4)
                if h < 0:
                    h += 360
                computed_hsv[i, j, :] = [h / 2, 255 * s, 255 * v]
        computed_hsv = computed_hsv.round().astype(np.uint8)
        print(in_img[0, 0, :])
        self.assert_img_equal(out_img, computed_hsv)

    def test_scale_size(self):
        assert im.Image.scale_size((300, 200), 0.5) == (150, 100)
        assert im.Image.scale_size((11, 22), 0.7) == (8, 15)

    def test_resize(self):
        img = im.Image.open(self.img_path)
        resized_img = img.resize((1000, 600))
        assert resized_img.size == (1000, 600)
        resized_img, w_scale, h_scale = img.resize((1000, 600), True)
        assert (resized_img.size == (1000, 600) and w_scale == 2.5
                and h_scale == 2.0)
        for mode in ['bilinear', 'bicubic', 'nearest', 'area', 'lanczos']:
            resized_img = img.resize((1000, 600), resample=mode)
            assert resized_img.size == (1000, 600)

    def test_resize_like(self):
        img = im.Image.open(self.img_path)
        a = np.zeros((100, 200, 3), dtype=np.uint8)
        resized_img = img.resize_like(a)
        assert resized_img.size == (200, 100)
        resized_img = img.resize_like(im.Image.fromarray(a))
        assert resized_img.size == (200, 100)
        with pytest.raises(TypeError):
            img.resize_like((200, 100))

    def test_resize_by_ratio(self):
        img = im.Image.open(self.img_path)
        resized_img = img.resize_by_ratio(1.5)
        assert resized_img.size == (600, 450)
        resized_img = img.resize_by_ratio(0.934)
        assert resized_img.size == (374, 280)

    def test_resize_keep_ar(self):
        img = im.Image.open(self.img_path)
        # resize (400, 300) to (max_1000, max_600)
        resized_img = img.resize_keep_ar(1000, 600)
        assert resized_img.size == (800, 600)
        resized_img, scale = img.resize_keep_ar(1000, 600, True)
        assert resized_img.size == (800, 600) and scale == 2.0
        # resize (400, 300) to (max_200, max_180)
        resized_img = img.resize_keep_ar(200, 180)
        assert resized_img.size == (200, 150)
        resized_img, scale = img.resize_keep_ar(200, 180, True)
        assert resized_img.size == (200, 150) and scale == 0.5
        # max_long_edge cannot be less than max_short_edge
        with pytest.raises(ValueError):
            img.resize_keep_ar(500, 600)

    def test_limit_size(self):
        img = im.Image.open(self.img_path)
        # limit to 800
        resized_img = img.limit_size(800)
        assert resized_img.size == (400, 300)
        resized_img, scale = img.limit_size(800, True)
        assert resized_img.size == (400, 300) and scale == 1
        # limit to 200
        resized_img = img.limit_size(200)
        assert resized_img.size == (200, 150)
        resized_img, scale = img.limit_size(200, True)
        assert resized_img.size == (200, 150) and scale == 0.5

    def test_flip(self):
        img = im.Image.open(self.img_path)
        f_img = img.flip()
        assert_array_equal(f_img.numpy(), img.numpy()[:, ::-1, :])
        f_img = img.flip('horizontal')
        assert_array_equal(f_img.numpy(), img.numpy()[:, ::-1, :])
        f_img = img.flip('vertical')
        assert_array_equal(f_img.numpy(), img.numpy()[::-1, :, :])

    def test_crop(self):
        img = im.Image.open(self.img_path)
        # yapf: disable
        bboxes = np.array([[100, 100, 199, 199],  # center
                           [0, 0, 150, 100],  # left-top corner
                           [250, 200, 399, 299],  # right-bottom corner
                           [0, 100, 399, 199],  # wide
                           [150, 0, 299, 299]])  # tall
        # yapf: enable
        # crop one bbox
        patch = img.crop(bboxes[0, :])
        assert patch.size == (100, 100)
        patch_path = path.join(path.dirname(__file__), 'data/patches')
        ref_patch = np.load(patch_path + '/0.npy')
        self.assert_img_equal(patch.numpy(), ref_patch)
        patches = img.crop(bboxes[[0], :])
        assert isinstance(patches, list) and len(patches) == 1
        self.assert_img_equal(patches[0].numpy(), ref_patch)
        # crop with no scaling and padding
        patches = img.crop(bboxes)
        assert len(patches) == bboxes.shape[0]
        for i in range(len(patches)):
            ref_patch = np.load(patch_path + '/{}.npy'.format(i))
            self.assert_img_equal(patches[i].numpy(), ref_patch)
        # crop with scaling and no padding
        patches = img.crop(bboxes, 1.2)
        for i in range(len(patches)):
            ref_patch = np.load(patch_path + '/scale_{}.npy'.format(i))
            self.assert_img_equal(patches[i].numpy(), ref_patch)
        # crop with scaling and padding
        patches = img.crop(bboxes, 1.2, pad_fill=(255, 255, 0))
        for i in range(len(patches)):
            ref_patch = np.load(patch_path + '/pad_{}.npy'.format(i))
            self.assert_img_equal(patches[i].numpy(), ref_patch)
        patches = img.crop(bboxes, 1.2, pad_fill=0)
        for i in range(len(patches)):
            ref_patch = np.load(patch_path + '/pad0_{}.npy'.format(i))
            self.assert_img_equal(patches[i].numpy(), ref_patch)

    def test_pad(self):
        data = np.random.randint(256, size=(10, 10, 3), dtype='uint8')
        img = im.Image.fromarray(data)
        # pad single value
        padded_img = img.pad((15, 12), 0)
        assert padded_img.size == (15, 12)
        assert_array_equal(img.numpy(), padded_img.numpy()[:10, :10, :])
        assert_array_equal(
            np.zeros((12, 5, 3), dtype='uint8'), padded_img.numpy()[:, 10:, :])
        assert_array_equal(
            np.zeros((2, 15, 3), dtype='uint8'), padded_img.numpy()[10:, :, :])
        data = np.random.randint(256, size=(10, 10, 3), dtype='uint8')
        img = im.Image.fromarray(data)
        # pad different values for different channels
        padded_img = img.pad((15, 12), (100, 110, 120))
        assert padded_img.size == (15, 12)
        assert_array_equal(img.numpy(), padded_img.numpy()[:10, :10, :])
        assert_array_equal(
            np.array([100, 110, 120], dtype='uint8') * np.ones(
                (12, 5, 3), dtype='uint8'), padded_img.numpy()[:, 10:, :])
        assert_array_equal(
            np.array([100, 110, 120], dtype='uint8') * np.ones(
                (2, 15, 3), dtype='uint8'), padded_img.numpy()[10:, :, :])
        with pytest.raises(AssertionError):
            img.pad((5, 5), 0)
        with pytest.raises(AssertionError):
            img.pad((5, 5), (0, 1))
