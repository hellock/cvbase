import numpy as np
from cvbase.det import bbox_overlaps, bbox_transform
from numpy.testing import assert_array_almost_equal


class TestDet(object):

    def test_bbox_overlaps(self):
        proposals = np.array([[100, 120, 300, 240], [0, 20, 20, 100]])
        gt = np.array([[125, 90, 280, 200], [50, 200, 280, 250]])
        ious = np.array([[0.43570912, 0.25874272], [0, 0]])
        assert_array_almost_equal(bbox_overlaps(proposals, gt), ious)

    def test_bbox_transform(self):
        proposals = np.array([[100, 120, 300, 240], [0, 20, 20, 100]])
        gt = np.array([[125, 90, 280, 200]])
        deltas = np.array([[0.012438, -0.289256, -0.253449, -0.08626],
                           [9.166667, 1.049383, 2.005334, 0.315081]])
        assert_array_almost_equal(bbox_transform(proposals, gt), deltas)
