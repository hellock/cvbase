import numpy as np
from cvbase import (bbox_overlaps, bbox_transform, bbox_transform_inv,
                    bbox_clip, bbox_flip, bbox_normalize, bbox_denormalize)
from numpy.testing import assert_array_almost_equal


class TestBboxTransform(object):

    def _bbox_tf_2d(self, proposals, gt):
        px = (proposals[:, 0] + proposals[:, 2]) * 0.5
        py = (proposals[:, 1] + proposals[:, 3]) * 0.5
        pw = (proposals[:, 2] - proposals[:, 0] + 1.0)
        ph = (proposals[:, 3] - proposals[:, 1] + 1.0)
        gx = (gt[:, 0] + gt[:, 2]) * 0.5
        gy = (gt[:, 1] + gt[:, 3]) * 0.5
        gw = (gt[:, 2] - gt[:, 0] + 1.0)
        gh = (gt[:, 3] - gt[:, 1] + 1.0)
        deltas = np.zeros((proposals.shape))
        for i in range(deltas.shape[0]):
            deltas[i, 0] = (gx[0] - px[i]) / pw[i]
            deltas[i, 1] = (gy[0] - py[i]) / ph[i]
            deltas[i, 2] = np.log(gw[0] / pw[i])
            deltas[i, 3] = np.log(gh[0] / ph[i])
        return deltas

    def _check_bbox_transform(self, proposals, gt):
        assert_array_almost_equal(
            bbox_transform(proposals, gt), self._bbox_tf_2d(proposals, gt))

    def test_bbox_transform(self):
        # proposals: (2, 4), gt: (1, 4)
        proposals1_2x4 = np.array([[100, 120, 300, 240], [0, 20, 20, 100]])
        gt1_1x4 = np.array([[125, 90, 280, 200]])
        deltas1 = np.array([[0.012438, -0.289256, -0.253449, -0.08626],
                            [9.166667, 1.049383, 2.005334, 0.315081]])
        assert_array_almost_equal(deltas1,
                                  self._bbox_tf_2d(proposals1_2x4, gt1_1x4))
        self._check_bbox_transform(proposals1_2x4, gt1_1x4)
        # proposals: (2, 4), gt: (2, 4)
        gt1_2x4 = np.tile(gt1_1x4, (2, 1))
        self._check_bbox_transform(proposals1_2x4, gt1_2x4)
        # proposals: (2, 4), gt: (1, 4)
        proposals2_2x4 = np.array([[50, 40, 120, 140], [200, 20, 210, 190]])
        gt2_1x4 = np.array([[60, 190, 130, 270]])
        gt2_2x4 = np.tile(gt2_1x4, (2, 1))
        self._check_bbox_transform(proposals2_2x4, gt2_1x4)
        # proposals: (1, 4), gt: (1, 4)
        proposals3_1x4 = np.array([[100, 120, 300, 240]])
        gt3_1x4 = np.array([[200, 50, 300, 170]])
        self._check_bbox_transform(proposals3_1x4, gt3_1x4)
        # proposals: (2, 2, 4), gt: (2, 2, 4)
        proposals4_2x2x4 = np.array([proposals1_2x4, proposals2_2x4])
        gt4_2x2x4 = np.array([gt1_2x4, gt2_2x4])
        deltas4_2x2x4 = bbox_transform(proposals4_2x2x4, gt4_2x2x4)
        assert_array_almost_equal(deltas4_2x2x4[1, ...],
                                  self._bbox_tf_2d(proposals2_2x4, gt2_2x4))
        # proposals: (2, 1, 4), gt: (2, 1, 4)
        proposals5_2x1x4 = proposals4_2x2x4[:, [0], :]
        gt5_2x1x4 = gt4_2x2x4[:, [0], :]
        deltas5_2x1x4 = bbox_transform(proposals5_2x1x4, gt5_2x1x4)
        assert_array_almost_equal(deltas5_2x1x4, deltas4_2x2x4[:, [0], :])

    def test_bbox_transform_inv(self):
        # proposals: (1, 4), deltas: (1, 8)
        proposals1_1x4 = np.array([[100, 120, 300, 240]])
        gt1_1x8 = np.array([[125, 90, 280, 200, 200, 50, 300, 170]])
        deltas1_1x8 = np.array([[
            0.01243781, -0.28925619, -0.2534489, -0.08626036,
            0.24875621, -0.57851237, -0.68818444, 0.
        ]])  # yapf: disable
        assert_array_almost_equal(
            bbox_transform_inv(proposals1_1x4, deltas1_1x8),
            gt1_1x8,
            decimal=5)
        # proposals: (2, 4), deltas: (2, 8)
        proposals2_2x4 = np.array([[100, 120, 300, 240], [150, 50, 190, 200]])
        gt2_2x8 = np.vstack((gt1_1x8, gt1_1x8))
        deltas2_1 = np.array([[
            0.79268295, 0.13245033, 1.33628392, -0.30774966,
            1.95121956, -0.09933775, 0.90154845, -0.22148931
        ]])  # yapf: disable
        deltas2_2x8 = np.vstack((deltas1_1x8, deltas2_1))
        assert_array_almost_equal(
            bbox_transform_inv(proposals2_2x4, deltas2_2x8),
            gt2_2x8,
            decimal=5)
        # proposals: (2, 1, 4), deltas: (2, 1, 8)
        proposals3_2x1x4 = proposals2_2x4[..., np.newaxis].transpose((0, 2, 1))
        gt3_2x1x8 = gt2_2x8[..., np.newaxis].transpose((0, 2, 1))
        deltas3_2x1x8 = deltas2_2x8[..., np.newaxis].transpose((0, 2, 1))
        assert_array_almost_equal(
            bbox_transform_inv(proposals3_2x1x4, deltas3_2x1x8),
            gt3_2x1x8,
            decimal=5)
        # proposals: (1, 2, 4), deltas: (1, 2, 8)
        proposals4_1x2x4 = proposals2_2x4[np.newaxis, ...]
        gt4_1x2x8 = gt2_2x8[np.newaxis, ...]
        deltas4_1x2x8 = deltas2_2x8[np.newaxis, ...]
        assert_array_almost_equal(
            bbox_transform_inv(proposals4_1x2x4, deltas4_1x2x8),
            gt4_1x2x8,
            decimal=5)

    def test_bboxes_clip(self):
        img_size = (768, 1024)
        # bbox of all valid values
        bbox1 = np.array([50, 100, 900, 700])
        gt1 = np.array([50, 100, 900, 700])
        # bbox of partial invalid values
        bbox2 = np.array([-10, 20, 50, 800])
        gt2 = np.array([0, 20, 50, 767])
        # bbox of all invalid values
        bbox3 = np.array([-100000, -1000, 2000, 900])
        gt3 = np.array([0, 0, 1023, 767])
        bboxes = np.array([bbox1, bbox2, bbox3])
        gts = np.array([gt1, gt2, gt3])
        assert_array_almost_equal(bbox_clip(bboxes, img_size), gts)

    def test_bboxes_flip(self):
        img_size = (768, 1024)
        # (1, 4)
        bboxes1 = np.array([[50, 100, 900, 700]])
        flipped1 = np.array([[123, 100, 973, 700]])
        assert_array_almost_equal(bbox_flip(bboxes1, img_size), flipped1)
        # (2, 4)
        bboxes2 = np.tile(bboxes1, (2, 1))
        flipped2 = np.tile(flipped1, (2, 1))
        assert_array_almost_equal(bbox_flip(bboxes2, img_size), flipped2)
        # (2, 8)
        bboxes3 = np.tile(bboxes2, (1, 2))
        flipped3 = np.tile(flipped2, (1, 2))
        assert_array_almost_equal(bbox_flip(bboxes3, img_size), flipped3)
        # (1, 2, 8)
        bboxes4 = bboxes3[np.newaxis, ...]
        flipped4 = flipped3[np.newaxis, ...]
        assert_array_almost_equal(bbox_flip(bboxes4, img_size), flipped4)

    def test_bboxes_normalize(self):
        means = [0.05, -0.05, -0.1, 0.1]
        stds = [0.1, 0.15, 0.2, 0.25]
        # deltas: (1, 4), means: (4, )
        deltas1 = np.array([[0.12, 0.13, -0.18, 0.05]])
        norm_deltas1 = np.array([[0.7, 1.2, -0.4, -0.2]])
        assert_array_almost_equal(
            bbox_normalize(deltas1, means, stds), norm_deltas1)
        # deltas: (2, 4), means: (4, )
        means = np.array(means)
        stds = np.array(stds)
        deltas2 = np.tile(deltas1, (2, 1))
        norm_deltas2 = np.tile(norm_deltas1, (2, 1))
        assert_array_almost_equal(
            bbox_normalize(deltas2, means, stds), norm_deltas2)
        # deltas: (2, 8), means: (4, )
        deltas3 = np.tile(deltas2, (1, 2))
        norm_deltas3 = np.tile(norm_deltas2, (1, 2))
        assert_array_almost_equal(
            bbox_normalize(deltas3, means, stds), norm_deltas3)
        # deltas: (2, 8), means: (8, )
        deltas4 = deltas3
        means = np.hstack((means, means))
        stds = np.hstack((stds, 2 * stds))
        norm_deltas4 = norm_deltas3
        norm_deltas4[:, 4:8] = norm_deltas4[:, 4:8] / 2
        assert_array_almost_equal(
            bbox_normalize(deltas4, means, stds), norm_deltas4)
        # deltas: (3, 2, 8), means: (8, )
        deltas5 = np.stack((deltas4, deltas4, deltas4))
        norm_deltas5 = np.stack((norm_deltas4, norm_deltas4, norm_deltas4))
        assert_array_almost_equal(
            bbox_normalize(deltas5, means, stds), norm_deltas5)

    def test_bboxes_denormalize(self):
        means = [0.05, -0.05, -0.1, 0.1]
        stds = [0.1, 0.15, 0.2, 0.25]
        # deltas: (1, 4), means: (4, )
        deltas1 = np.array([[0.12, 0.13, -0.18, 0.05]])
        norm_deltas1 = np.array([[0.7, 1.2, -0.4, -0.2]])
        assert_array_almost_equal(
            bbox_denormalize(norm_deltas1, means, stds), deltas1)
        # deltas: (2, 4), means: (4, )
        means = np.array(means)
        stds = np.array(stds)
        deltas2 = np.tile(deltas1, (2, 1))
        norm_deltas2 = np.tile(norm_deltas1, (2, 1))
        assert_array_almost_equal(
            bbox_denormalize(norm_deltas2, means, stds), deltas2)
        # deltas: (2, 8), means: (4, )
        deltas3 = np.tile(deltas2, (1, 2))
        norm_deltas3 = np.tile(norm_deltas2, (1, 2))
        assert_array_almost_equal(
            bbox_denormalize(norm_deltas3, means, stds), deltas3)
        # deltas: (2, 8), means: (8, )
        deltas4 = deltas3
        means = np.hstack((means, means))
        stds = np.hstack((stds, 2 * stds))
        norm_deltas4 = norm_deltas3
        norm_deltas4[:, 4:8] = norm_deltas4[:, 4:8] / 2
        assert_array_almost_equal(
            bbox_denormalize(norm_deltas4, means, stds), deltas4)
        # deltas: (2, 2, 8), means: (8, )
        deltas5 = np.stack((deltas4, deltas4))
        norm_deltas5 = np.stack((norm_deltas4, norm_deltas4))
        assert_array_almost_equal(
            bbox_denormalize(norm_deltas5, means, stds), deltas5)


class TestEval(object):

    def test_bbox_overlaps(self):
        bbox1 = np.array([[50, 50, 100, 100]])
        bbox2 = np.array([[75, 75, 150, 150],
                          [100, 100, 150, 150],
                          [150, 150, 200, 200]])  # yapf: disable
        result = np.array([[0.08778081, 0.00019227, 0.]])
        assert_array_almost_equal(bbox_overlaps(bbox1, bbox2), result)
