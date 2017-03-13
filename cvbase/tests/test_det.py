import numpy as np
from cvbase.det import bbox_overlaps, bbox_transform, bbox_transform_inv
from numpy.testing import assert_array_almost_equal


class TestDet(object):

    def test_bbox_overlaps(self):
        proposals1 = np.array([[100, 120, 300, 240], [0, 20, 20, 100]])
        gt1 = np.array([[125, 90, 280, 200], [50, 200, 280, 250]])
        ious1 = np.array([[0.43570912, 0.25874272], [0, 0]])
        assert_array_almost_equal(bbox_overlaps(proposals1, gt1), ious1)

    def naive_bbox_transform_2D(self, proposals, gt):
        px = (proposals[:,0] + proposals[:,2])*0.5
        py = (proposals[:,1] + proposals[:,3])*0.5
        pw = (proposals[:,2] - proposals[:,0] + 1.0)
        ph = (proposals[:,3] - proposals[:,1] + 1.0)
        gx = (gt[:,0] + gt[:,2])*0.5
        gy = (gt[:,1] + gt[:,3])*0.5
        gw = (gt[:,2] - gt[:,0] + 1.0)
        gh = (gt[:,3] - gt[:,1] + 1.0)
        deltas = np.zeros((proposals.shape))
        for i in range(deltas.shape[0]):
            deltas[i,0] = (gx[0] - px[i])/pw[i]
            deltas[i,1] = (gy[0] - py[i])/ph[i]
            deltas[i,2] = np.log(gw[0]/pw[i])
            deltas[i,3] = np.log(gh[0]/ph[i])
        return deltas

    def test_bbox_transform(self):
        # test 1
        proposals1 = np.array([[100, 120, 300, 240], [0, 20, 20, 100]])
        gt1 = np.array([[125, 90, 280, 200]])
        deltas1 = np.array([[0.012438, -0.289256, -0.253449, -0.08626],
                           [9.166667, 1.049383, 2.005334, 0.315081]])
        assert_array_almost_equal(deltas1, self.naive_bbox_transform_2D(proposals1, gt1))
        assert_array_almost_equal(bbox_transform(proposals1, gt1), self.naive_bbox_transform_2D(proposals1, gt1))
        # test 2
        gt2 = np.array([[125, 90, 280, 200], [125, 90, 280, 200]])
        assert_array_almost_equal(bbox_transform(proposals1, gt1), self.naive_bbox_transform_2D(proposals1, gt2))
        # test3 
        proposals2 = np.array([[50, 40, 120, 140], [200,20, 210, 190]])
        gt2 = np.array([[60, 190, 130, 270]])
        deltas2 = self.naive_bbox_transform_2D(proposals2, gt2)
        assert_array_almost_equal(bbox_transform(proposals2, gt2), self.naive_bbox_transform_2D(proposals2, gt2))
        # test 4
        proposals3 = np.array([proposals1, proposals2])
        gt1 = np.array([[125, 90, 280, 200], [125, 90, 280, 200]])
        gt2 = np.array([[60, 190, 130, 270], [60, 190, 130, 270]])
        gt3 = np.array([gt1, gt2])
        deltas3 = np.array([deltas1, deltas2])
        assert_array_almost_equal(bbox_transform(proposals3, gt3), deltas3)
        # test 5
        proposals4 = np.array([proposals1[0], proposals2[1]])
        gt4 = np.array([gt1[0], gt2[0]])
        deltas4 = np.array([deltas1[0], deltas2[1]])
        assert_array_almost_equal(bbox_transform(proposals4, gt4), deltas4)
        # test 6 it seams this function doesn't allow input with single proposal and multi gt_bboxes
        proposals4 = np.array([proposals1[0]])
        gt4 = np.array([gt1[0], gt1[0]])
        deltas4 = np.array([deltas1[0], deltas1[0]])
        #assert_array_almost_equal(bbox_transform(proposals4, gt4), deltas4)

    def test_bbox_transform_inv(self):
        # test 1
        proposals1 = np.array([[100, 120, 300, 240]])
        gt1_0 = np.array([[125, 90, 280, 200]])
        gt1_1 = np.array([[200, 50, 300, 170]])
        deltas1_gt = np.array([[0.012438, -0.289256, -0.253449, -0.08626]])
        gt1 = np.array([gt1_0[0], gt1_1[0]])
        deltas1_0 = bbox_transform(proposals1, gt1_0)
        deltas1_1 = bbox_transform(proposals1, gt1_1)
        deltas1 = np.hstack((deltas1_0[0], deltas1_1[0]))
        assert_array_almost_equal(deltas1_gt, deltas1_0)
        assert_array_almost_equal(bbox_transform_inv(proposals1, deltas1_0), gt1_0)
