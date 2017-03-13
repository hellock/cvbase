import numpy as np
from cvbase.det import (bbox_overlaps, bbox_transform, bbox_transform_inv,
                        clip_bboxes)
from numpy.testing import assert_array_almost_equal


class TestDet(object):

    def test_bbox_overlaps(self):
        bbox1 = np.array([[50, 50, 100, 100]])
        bbox2 = np.array([[75, 75, 150, 150],
                          [100, 100, 150, 150],
                          [150, 150, 200, 200]])  # yapf: disable
        result = np.array([[0.08778081, 0.00019227, 0.]])
        assert_array_almost_equal(bbox_overlaps(bbox1, bbox2), result)

    def _naive_bbox_transform_2d(self, proposals, gt):
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
            bbox_transform(proposals, gt),
            self._naive_bbox_transform_2d(proposals, gt))

    def test_bbox_transform(self):
        # proposals: (2, 4), gt: (1, 4)
        proposals1_2x4 = np.array([[100, 120, 300, 240], [0, 20, 20, 100]])
        gt1_1x4 = np.array([[125, 90, 280, 200]])
        deltas1 = np.array([[0.012438, -0.289256, -0.253449, -0.08626],
                            [9.166667, 1.049383, 2.005334, 0.315081]])
        assert_array_almost_equal(
            deltas1, self._naive_bbox_transform_2d(proposals1_2x4, gt1_1x4))
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
                                  self._naive_bbox_transform_2d(proposals2_2x4,
                                                                gt2_2x4))
        # proposals: (2, 1, 4), gt: (2, 1, 4)
        proposals5_2x1x4 = proposals4_2x2x4[:, [0], :]
        gt5_2x1x4 = gt4_2x2x4[:, [0], :]
        deltas5_2x1x4 = bbox_transform(proposals5_2x1x4, gt5_2x1x4)
        assert_array_almost_equal(deltas5_2x1x4, deltas4_2x2x4[:, [0], :])

    def test_bbox_transform_inv(self):
        # test 1 with 1 proposal 2 gt
        proposals1 = np.array([[100, 120, 300, 240]])
        gt1_0 = np.array([[125, 90, 280, 200]])
        gt1_1 = np.array([[200, 50, 300, 170]])
        deltas1_gt_0 = np.array(
            [[0.01243781, -0.28925619, -0.2534489, -0.08626036]])
        deltas1_gt_1 = np.array([[0.24875621, -0.57851237, -0.68818444, 0.]])
        gt1 = np.hstack((gt1_0, gt1_1))
        deltas1 = np.hstack((deltas1_gt_0, deltas1_gt_1))
        assert_array_almost_equal(
            bbox_transform_inv(proposals1, deltas1),
            gt1.astype(np.float32),
            decimal=5)
        # test 2 with 2 proposal 2 gt
        proposals1 = np.array([[100, 120, 300, 240], [150, 50, 190, 200]])
        gt1_0 = np.array([[125, 90, 280, 200]])
        gt1_1 = np.array([[200, 50, 300, 170]])
        gt1 = np.array(
            [np.hstack((gt1_0[0], gt1_1[0])), np.hstack((gt1_0[0], gt1_1[0]))])
        # deltas1_0 = bbox_transform(proposals1, gt1_0)
        # deltas1_1 = bbox_transform(proposals1, gt1_1)
        deltas1_0 = np.array(
            [[0.01243781, -0.28925619, -0.2534489, -0.08626036],
             [0.79268295, 0.13245033, 1.33628392, -0.30774966]])
        deltas1_1 = np.array(
            [[0.24875621, -0.57851237, -0.68818444, 0.],
             [1.95121956, -0.09933775, 0.90154845, -0.22148931]])
        deltas1 = np.array([
            np.hstack((deltas1_0[0], deltas1_1[0])), np.hstack(
                (deltas1_0[1], deltas1_1[1]))
        ])
        assert_array_almost_equal(
            bbox_transform_inv(proposals1, deltas1),
            gt1.astype(np.float32),
            decimal=5)
        # test 3 3D data shape = (2, 1, 4)
        proposals1 = np.array([[[100, 120, 300, 240]], [[150, 50, 190, 200]]])
        gt1_0 = np.array([[[125, 90, 280, 200]]])
        gt1_1 = np.array([[[200, 50, 300, 170]]])
        gt1 = np.array(
            [np.hstack((gt1_0[0], gt1_1[0])), np.hstack((gt1_0[0], gt1_1[0]))])
        # deltas1_0 = bbox_transform(proposals1, gt1_0)
        # deltas1_1 = bbox_transform(proposals1, gt1_1)
        deltas1_0 = np.array(
            [[[0.01243781, -0.28925619, -0.2534489, -0.08626036]],
             [[0.79268295, 0.13245033, 1.33628392, -0.30774966]]])
        deltas1_1 = np.array(
            [[[0.24875621, -0.57851237, -0.68818444, 0.]],
             [[1.95121956, -0.09933775, 0.90154845, -0.22148931]]])
        deltas1 = np.array([
            np.hstack((deltas1_0[0], deltas1_1[0])), np.hstack(
                (deltas1_0[1], deltas1_1[1]))
        ])
        assert_array_almost_equal(
            bbox_transform_inv(proposals1, deltas1),
            gt1.astype(np.float32),
            decimal=5)
        # test 3 3D data shape = (1, 2, 4)
        proposals1 = np.array([[[100, 120, 300, 240], [150, 50, 190, 200]]])
        gt1_0 = np.array([[[125, 90, 280, 200]]])
        gt1_1 = np.array([[[200, 50, 300, 170]]])
        gt1 = np.array([[
            np.hstack((gt1_0[0][0], gt1_1[0][0])), np.hstack((gt1_0[0][0],
                                                              gt1_1[0][0]))
        ]])
        # deltas1_0 = bbox_transform(proposals1, gt1_0)
        # deltas1_1 = bbox_transform(proposals1, gt1_1)
        deltas1_0 = np.array(
            [[[0.01243781, -0.28925619, -0.2534489, -0.08626036],
              [0.79268295, 0.13245033, 1.33628392, -0.30774966]]])
        deltas1_1 = np.array(
            [[[0.24875621, -0.57851237, -0.68818444, 0.],
              [1.95121956, -0.09933775, 0.90154845, -0.22148931]]])
        deltas1 = np.array([
            np.hstack((deltas1_0[0][0], deltas1_1[0][0])), np.hstack(
                (deltas1_0[0][1], deltas1_1[0][1]))
        ])
        assert_array_almost_equal(
            bbox_transform_inv(proposals1, deltas1),
            gt1.astype(np.float32),
            decimal=5)

    def test_clip_bboxes(self):
        image_size = (768, 1024)
        # bbox with all valid inputs
        bbox1 = np.array([50, 100, 900, 700])
        gt1 = np.array([50, 100, 900, 700])
        # bbox with some part of invalid inputs
        bbox2 = np.array([-10, 20, 50, 800])
        gt2 = np.array([0, 20, 50, 767])
        # bbox with all invalid inputs
        bbox3 = np.array([-100000, -1000, 2000, 900])
        gt3 = np.array([0, 0, 1023, 767])
        bboxes = np.array([bbox1, bbox2, bbox3])
        gts = np.array([gt1, gt2, gt3])
        assert_array_almost_equal(clip_bboxes(bboxes, image_size), gts)
