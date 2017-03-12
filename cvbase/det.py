import numpy as np


def bbox_transform(proposals, gt):
    """calculate regression deltas from proposals and ground truths

    dx = (gx - px) / pw, dw = log(gw / pw)

    Args:
        proposals(ndarray): shape (..., 4)
        gt(ndarray): shape (..., 4) or (1.., 4)
    Output:
        ndarray: same shape as proposals
    """
    assert proposals.ndim == gt.ndim
    if gt.shape[0] == 1:
        shape = [1 for _ in range(proposals.ndim)]
        shape[0] = proposals.shape[0]
        gt = np.tile(gt, tuple(shape))
    assert proposals.shape == gt.shape
    proposals = proposals.astype(np.float32)
    gt = gt.astype(np.float32)
    px = (proposals[..., 0] + proposals[..., 2]) * 0.5  # px
    py = (proposals[..., 1] + proposals[..., 3]) * 0.5  # py
    pw = proposals[..., 2] - proposals[..., 0] + 1.0  # pw
    ph = proposals[..., 3] - proposals[..., 1] + 1.0  # ph

    gx = (gt[..., 0] + gt[..., 2]) * 0.5  # gx
    gy = (gt[..., 1] + gt[..., 3]) * 0.5  # gy
    gw = gt[..., 2] - gt[..., 0] + 1.0  # gw
    gh = gt[..., 3] - gt[..., 1] + 1.0  # gh

    tx = (gx - px) / pw
    ty = (gy - py) / ph
    tw = np.log(gw / pw)
    th = np.log(gh / ph)
    deltas = np.concatenate(
        (tx[..., np.newaxis], ty[..., np.newaxis], tw[..., np.newaxis],
         th[..., np.newaxis]),
        axis=-1)
    return deltas


def bbox_transform_inv(bboxes, deltas):
    """get prediction bboxes from input bboxes and deltas

    pw = gw * exp(dw), px = gx + dx * pw

    Args:
        bboxes(ndarray): shape (..., 4) [x1, y1, x2, y2]
        deltas(ndarray): shape (..., 4*k) [dx, dy, dw, dh]
    Output:
        ndarray: same shape as input bboxes
    """
    assert bboxes.shape[-1] == 4
    assert deltas.shape[-1] % 4 == 0
    gx = (bboxes[..., 0] + bboxes[..., 2]) * 0.5
    gy = (bboxes[..., 1] + bboxes[..., 3]) * 0.5
    gw = bboxes[..., 2] - bboxes[..., 0] + 1.0
    gh = bboxes[..., 3] - bboxes[..., 1] + 1.0
    pw = gw[..., np.newaxis] * np.exp(deltas[..., 2::4])
    ph = gh[..., np.newaxis] * np.exp(deltas[..., 3::4])
    px = gx[..., np.newaxis] + deltas[..., 0::4] * pw
    py = gy[..., np.newaxis] + deltas[..., 1::4] * ph
    shape = list(px.shape)
    shape[-1] = shape[-1] * 4
    return np.stack(
        (px - pw * 0.5, py - ph * 0.5, px + pw * 0.5, py + ph * 0.5),
        axis=-1).reshape(tuple(shape))


def clip_bboxes(bboxes, img_shape):
    """limit bboxes to fit the image size

    Args:
        bboxes(ndarray): shape (..., 4*k)
        img_shape(tuple): (height, width)
    """
    assert bboxes.shape[-1] % 4 == 0
    bboxes[..., 0::4] = np.maximum(
        np.minimum(bboxes[..., 0::4], img_shape[1] - 1), 0)
    bboxes[..., 1::4] = np.maximum(
        np.minimum(bboxes[..., 1::4], img_shape[0] - 1), 0)
    bboxes[..., 2::4] = np.maximum(
        np.minimum(bboxes[..., 2::4], img_shape[1] - 1), 0)
    bboxes[..., 3::4] = np.maximum(
        np.minimum(bboxes[..., 3::4], img_shape[0] - 1), 0)
    return bboxes


def bbox_overlaps(bboxes1, bboxes2):
    """calculate the ious between each bbox of bboxes1 and bboxes2

    Args:
        bboxes1(ndarray): shape (n, 4)
        bboxes2(ndarray): shape (k, 4)
    Output:
        ious(ndarray): shape (n, k)
    """
    bboxes1 = bboxes1.astype(np.float32)
    bboxes2 = bboxes2.astype(np.float32)
    rows = bboxes1.shape[0]
    cols = bboxes2.shape[0]
    ious = np.zeros((rows, cols), dtype=np.float32)
    if rows * cols == 0:
        return ious
    for i in range(bboxes1.shape[0]):
        x_start = np.maximum(bboxes1[i, 0], bboxes2[:, 0])
        y_start = np.maximum(bboxes1[i, 1], bboxes2[:, 1])
        x_end = np.minimum(bboxes1[i, 2], bboxes2[:, 2])
        y_end = np.minimum(bboxes1[i, 3], bboxes2[:, 3])
        overlap = (np.maximum(x_end - x_start + 1, 0) *
                   np.maximum(y_end - y_start + 1, 0))
        area1 = (bboxes1[i, 2] - bboxes1[i, 0] + 1) * (
            bboxes1[i, 3] - bboxes1[i, 1] + 1)
        area2 = (bboxes2[:, 2] - bboxes2[:, 0] + 1) * (
            bboxes2[:, 3] - bboxes2[:, 1] + 1)
        union = area1 + area2 - overlap
        ious[i, :] = overlap / union
    return ious


def _recalls(all_ious, proposal_nums=None, thrs=None):

    if thrs is None:
        thrs = np.array([0.5])
    elif isinstance(thrs, list):
        thrs = np.array(thrs)
    elif isinstance(thrs, float):
        thrs = thrs * np.ones((1, ))

    img_num = all_ious.shape[0]
    total_gt_num = sum([ious.shape[0] for ious in all_ious])

    _ious = np.zeros((proposal_nums.size, total_gt_num), dtype=np.float32)
    for k, proposal_num in enumerate(proposal_nums):
        tmp_ious = np.zeros(0)
        for i in range(img_num):
            ious = all_ious[i][:, :proposal_num].copy()
            gt_ious = np.zeros((ious.shape[0]))
            for j in range(ious.shape[0]):
                gt_max_overlaps = ious.argmax(axis=1)
                max_ious = ious[np.arange(0, ious.shape[0]), gt_max_overlaps]
                gt_idx = max_ious.argmax()
                gt_ious[j] = max_ious[gt_idx]
                box_idx = gt_max_overlaps[gt_idx]
                ious[gt_idx, :] = -1
                ious[:, box_idx] = -1
            tmp_ious = np.hstack((tmp_ious, gt_ious))
        _ious[k, :] = tmp_ious

    _ious = np.fliplr(np.sort(_ious, axis=1))
    recalls = np.zeros((proposal_nums.size, thrs.size))
    for i, thr in enumerate(thrs):
        recalls[:, i] = (_ious >= thr).sum(axis=1) / float(total_gt_num)

    return recalls


def bbox_recalls(gts, proposals, proposal_nums=None, thrs=None):
    """calculate recalls
    Args:
        gts(ndarray): a numpy array of object type,
                      each element is a ndarray of shape (n, 4)
        proposals(ndarray): a numpy array of object type,
                            each element is a ndarray of shape (k, 4) or (k, 5)
        proposal_nums(int or list of int or ndarray): top N proposals
        thrs(float or list or ndarray): iou thresholds
    """

    img_num = gts.shape[0]
    assert (img_num == proposals.shape[0])

    if isinstance(proposal_nums, list):
        proposal_nums = np.array(proposal_nums)
    elif isinstance(proposal_nums, int):
        proposal_nums = np.array([proposal_nums])

    for i in range(img_num):
        if proposals[i].ndim == 2 and proposals[i].shape[1] == 5:
            scores = proposals[i][:, 4]
            sort_idx = np.argsort(scores)[::-1]
            proposals[i] = proposals[i][sort_idx, :]

    all_ious = []
    for i in range(img_num):
        prop_num = min(proposals[i].shape[0], proposal_nums[-1])
        if gts[i] is None or gts[i].shape[0] == 0:
            ious = np.zeros((0, proposals[i].shape[0]), dtype=np.float32)
        else:
            ious = bbox_overlaps(gts[i], proposals[i][:prop_num, :4])
        all_ious.append(ious)
    all_ious = np.array(all_ious)

    return _recalls(all_ious, proposal_nums, thrs)


def average_precision(recall, precision):
    """Calculate average precision
    """
    mrec = np.concatenate(([0.], recall, [1.]))
    mpre = np.concatenate(([0.], precision, [0.]))
    for i in range(mpre.shape[0] - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    ap = np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1])
    return ap


def eval_map(det_results, gt_bboxes, gt_labels, iou_thr=0.5):
    """Evaluate mAP of a dataset

    Args:
        det_results(list): a list of list, [[cls1, cls2], [cls1, cls2], ...]
        gt_bboxes(list): ground truth bboxes of each image
        gt_labels(list): ground truth labels of each image
        iou_thr(float): IoU threshold
        out_file(str or None): filename to save all precisions and recalls
    Output:
        tuple: (mAP, [dict, dict, ...])
    """
    eval_results = []
    cls_num = len(det_results[0])  # positive class num
    for i in range(cls_num):  # for each class
        dets = [det[i] for det in det_results]
        gts = [
            bbox[label == i + 1, :]
            for bbox, label in zip(gt_bboxes, gt_labels)
        ]
        gt_num = sum([gt.shape[0] for gt in gts])
        img_idxs = [
            i * np.ones(det.shape[0], dtype=np.int32)
            for i, det in enumerate(dets)
        ]
        dets = np.vstack(dets)
        img_idxs = np.concatenate(img_idxs)
        # sort all detections by scores in descending order
        sort_idx = np.argsort(dets[:, -1])[::-1]
        dets = dets[sort_idx, :]
        img_idxs = img_idxs[sort_idx]
        covered = [np.zeros(gt.shape[0], dtype=np.int32) for gt in gts]
        det_num = dets.shape[0]
        fp = np.zeros(det_num, dtype=np.float32)
        tp = np.zeros(det_num, dtype=np.float32)
        # for each det bbox, check if it is a true positive
        for j in range(det_num):
            img_idx = img_idxs[j]
            if gts[img_idx].shape[0] == 0:
                fp[j] = 1
                continue
            ious = bbox_overlaps(dets[np.newaxis, j, :], gts[img_idx])
            if ious.max() > iou_thr and covered[img_idx][ious.argmax()] == 0:
                covered[img_idx][ious.argmax()] = 1
                tp[j] = 1
            else:
                fp[j] = 1
        tp = np.cumsum(tp)
        fp = np.cumsum(fp)
        eps = np.finfo(np.float32).eps
        recall = tp / np.maximum(float(gt_num), eps)
        precision = tp / np.maximum((tp + fp), eps)
        ap = average_precision(recall, precision)
        eval_results.append({
            'gt_num': gt_num,
            'det_num': det_num,
            'recall': recall,
            'precision': precision,
            'ap': ap
        })
    aps = []
    for cls_result in eval_results:
        if cls_result['gt_num'] > 0:
            aps.append(cls_result['ap'])
    return np.array(aps).mean(), eval_results


def print_map_summary(mean_ap, results):
    """Print mAP and results of each class
    """
    print(50 * '-')
    for i, cls_result in enumerate(results):
        print('class {}, gt num: {}, det num: {}, recall: {:.4f}, ap: {:.4f}'.
              format(i + 1, cls_result['gt_num'], cls_result['det_num'],
                     cls_result['recall'][-1], cls_result['ap']))
    print('mAP: {:.4f}'.format(mean_ap))
    print(50 * '-')
