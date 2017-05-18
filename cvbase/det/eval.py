import numpy as np
import matplotlib.pyplot as plt
from six.moves import zip
from terminaltables import AsciiTable


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
    exchange = False
    if bboxes1.shape[0] > bboxes2.shape[0]:
        bboxes1, bboxes2 = bboxes2, bboxes1
        ious = np.zeros((cols, rows), dtype=np.float32)
        exchange = True
    area1 = (bboxes1[:, 2] - bboxes1[:, 0] + 1) * (
        bboxes1[:, 3] - bboxes1[:, 1] + 1)
    area2 = (bboxes2[:, 2] - bboxes2[:, 0] + 1) * (
        bboxes2[:, 3] - bboxes2[:, 1] + 1)
    for i in range(bboxes1.shape[0]):
        x_start = np.maximum(bboxes1[i, 0], bboxes2[:, 0])
        y_start = np.maximum(bboxes1[i, 1], bboxes2[:, 1])
        x_end = np.minimum(bboxes1[i, 2], bboxes2[:, 2])
        y_end = np.minimum(bboxes1[i, 3], bboxes2[:, 3])
        overlap = np.maximum(x_end - x_start + 1, 0) * np.maximum(
            y_end - y_start + 1, 0)
        union = area1[i] + area2 - overlap
        ious[i, :] = overlap / union
    if exchange:
        ious = ious.T
    return ious


def _recalls(all_ious, proposal_nums, thrs):

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


def set_recall_param(proposal_nums, iou_thrs):
    """Check proposal_nums and iou_thrs and set correct format
    """
    if isinstance(proposal_nums, list):
        _proposal_nums = np.array(proposal_nums)
    elif isinstance(proposal_nums, int):
        _proposal_nums = np.array([proposal_nums])
    else:
        _proposal_nums = proposal_nums

    if iou_thrs is None:
        _iou_thrs = np.array([0.5])
    elif isinstance(iou_thrs, list):
        _iou_thrs = np.array(iou_thrs)
    elif isinstance(iou_thrs, float):
        _iou_thrs = np.array([iou_thrs])
    else:
        _iou_thrs = iou_thrs

    return _proposal_nums, _iou_thrs


def bbox_recalls(gts,
                 proposals,
                 proposal_nums=None,
                 iou_thrs=None,
                 print_summary=True):
    """calculate recalls
    Args:
        gts(list or ndarray): a list of arrays of shape (n, 4)
        proposals(list or ndarray): a list of arrays of shape (k, 4) or (k, 5)
        proposal_nums(int or list of int or ndarray): top N proposals
        thrs(float or list or ndarray): iou thresholds
    """

    img_num = len(gts)
    assert img_num == len(proposals)

    proposal_nums, iou_thrs = set_recall_param(proposal_nums, iou_thrs)

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
    recalls = _recalls(all_ious, proposal_nums, iou_thrs)
    if print_summary:
        print_recall_summary(recalls, proposal_nums, iou_thrs)
    return recalls


def print_recall_summary(recalls,
                         proposal_nums,
                         iou_thrs,
                         row_idxs=None,
                         col_idxs=None):
    """Print recalls in a table

    Args:
        recalls(ndarray): calculated from `bbox_recalls`
        proposal_nums(ndarray): top N proposals
        iou_thrs(ndarray): iou thresholds
        row_idxs(ndarray): which rows(proposal nums) to print
        col_idxs(ndarray): which cols(iou thresholds) to print
    """
    if row_idxs is None:
        row_idxs = np.arange(proposal_nums.size)
    if col_idxs is None:
        col_idxs = np.arange(iou_thrs.size)
    row_header = [''] + iou_thrs[col_idxs].tolist()
    table_data = [row_header]
    for i, num in enumerate(proposal_nums[row_idxs]):
        row = [
            '{:.3f}'.format(val)
            for val in recalls[row_idxs[i], col_idxs].tolist()
        ]
        row.insert(0, num)
        table_data.append(row)
    table = AsciiTable(table_data)
    print(table.table)


def plot_num_recall(recalls, proposal_nums):
    """Plot Proposal_num-Recalls curve

    Args:
        recalls(ndarray or list): shape (k,)
        proposal_nums(ndarray or list): same shape as `recalls`
    """
    if isinstance(proposal_nums, np.ndarray):
        _proposal_nums = proposal_nums.tolist()
    else:
        _proposal_nums = proposal_nums
    if isinstance(recalls, np.ndarray):
        _recalls = recalls.tolist()
    else:
        _recalls = recalls

    f = plt.figure()
    plt.plot([0] + _proposal_nums, [0] + _recalls)
    plt.xlabel('Proposal num')
    plt.ylabel('Recall')
    plt.axis([0, proposal_nums.max(), 0, 1])
    f.show()


def plot_iou_recall(recalls, iou_thrs):
    """Plot IoU-Recalls curve

    Args:
        recalls(ndarray or list): shape (k,)
        iou_thrs(ndarray or list): same shape as `recalls`
    """
    if isinstance(iou_thrs, np.ndarray):
        _iou_thrs = iou_thrs.tolist()
    else:
        _iou_thrs = iou_thrs
    if isinstance(recalls, np.ndarray):
        _recalls = recalls.tolist()
    else:
        _recalls = recalls

    f = plt.figure()
    plt.plot(_iou_thrs + [1.0], _recalls + [0.])
    plt.xlabel('IoU')
    plt.ylabel('Recall')
    plt.axis([iou_thrs.min(), 1, 0, 1])
    f.show()


def average_precision(recall, precision, mode='area'):
    """Calculate average precision
    """
    if mode == 'area':
        mrec = np.concatenate(([0.], recall, [1.]))
        mpre = np.concatenate(([0.], precision, [0.]))
        for i in range(mpre.shape[0] - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])
        idx = np.where(mrec[1:] != mrec[:-1])[0]
        ap = np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1])
    elif mode == '11points':
        ap = 0.0
        for thr in np.arange(0, 1 + 1e-3, 0.1):
            precs = precision[recall >= thr]
            prec = precs.max() if precs.size > 0 else 0
            ap += prec
        ap /= 11
    else:
        raise ValueError(
            'Unrecognized mode, only "area" and "11points" are supported')
    return ap


def _tpfp_imagenet(det_bbox, gt_bboxes, gt_covered, default_iou_thr):
    """Check if a detected bbox is a true positive or false positive.

    Args:
        det_bbox(ndarray): the detected bbox
        gt_bboxes(ndarray): ground truth bboxes of this image
        gt_covered(ndarray): indicate if gts are matched
        default_iou_thr(float): the iou thresholds for medium and large bboxes
    Output:
        tuple: (tp, fp), either 0 or 1
    """
    bbox_max_iou = -1
    matched_gt = -1
    ious = bbox_overlaps(det_bbox[np.newaxis, :], gt_bboxes - 1)
    gt_w = gt_bboxes[:, 2] - gt_bboxes[:, 0] + 1
    gt_h = gt_bboxes[:, 3] - gt_bboxes[:, 1] + 1
    iou_thrs = np.minimum((gt_w * gt_h) / ((gt_w + 10.0) * (gt_h + 10.0)),
                          default_iou_thr)
    for k in range(ious.size):
        if gt_covered[k]:
            continue
        if ious[0, k] >= iou_thrs[k] and ious[0, k] > bbox_max_iou:
            bbox_max_iou = ious[0, k]
            matched_gt = k
    if matched_gt >= 0:
        gt_covered[matched_gt] = 1
        return 1, 0
    else:
        return 0, 1


def _tpfp_default(det_bbox, gt_bboxes, gt_covered, iou_thr, gt_difficults):
    """Check if a detected bbox is a true positive or false positive

    Args:
        det_bbox(ndarray): the detected bbox
        gt_bboxes(ndarray): ground truth bboxes of this image
        gt_covered(ndarray): indicate if gts are matched
        iou_thr(float): the iou thresholds
        gt_difficults(ndarray): indicate if gts are difficult or not
    Output:
        tuple: (tp, fp), either 0 or 1
    """
    ious = bbox_overlaps(det_bbox[np.newaxis, :], gt_bboxes)
    if ious.max() >= iou_thr:
        if not gt_difficults[ious.argmax()]:
            if gt_covered[ious.argmax()] == 0:
                gt_covered[ious.argmax()] = 1
                return 1, 0
            else:
                return 0, 1
        else:
            return 0, 0  # ignore this detected bbox
    else:
        return 0, 1


def eval_map(det_results,
             gt_bboxes,
             gt_labels,
             iou_thr=0.5,
             dataset='voc12',
             print_summary=True):
    """Evaluate mAP of a dataset

    Args:
        det_results(list): a list of list, [[cls1_det, cls2_det, ...], ...]
        gt_bboxes(list): ground truth bboxes of each image, a list of K*4 array
        gt_labels(list): ground truth labels of each image, a list of K/K*2 array
        iou_thr(float): IoU threshold
        print_summary(bool): whether to print the mAP summary
        dataset(str): dataset name, there are minor differences in metrics
                      for different datsets, e.g. "voc07", "voc12", "imagenet"
    Output:
        tuple: (mAP, [dict, dict, ...])
    """
    eval_results = []
    cls_num = len(det_results[0])  # positive class num
    for i in range(cls_num):  # for each class
        dets = [det[i] for det in det_results]
        gts = []  # gt bboxes of this class
        gt_difficult = []  # difficult indicator of this class
        labels = [
            label[:, 0] if label.ndim == 2 and label.shape[1] == 1 else label
            for label in gt_labels
        ]
        for bbox, label in zip(gt_bboxes, labels):
            cls_idx = (label == i + 1
                       if label.ndim == 1 else label[:, 0] == i + 1)
            gt = bbox[cls_idx, :] if bbox.shape[0] > 0 else bbox
            difficult = label[cls_idx, 1] if label.ndim > 1 else np.zeros(
                gt.shape[0], dtype=np.int32)
            gts.append(gt)
            gt_difficult.append(difficult)

        gt_num = sum([bbox.shape[0] for bbox in gts]) - sum(
            [diff.sum() for diff in gt_difficult])
        img_idxs = [
            k * np.ones(det.shape[0], dtype=np.int32)
            for k, det in enumerate(dets)
        ]

        dets = np.vstack(dets)
        img_idxs = np.concatenate(img_idxs)
        # sort all detections by scores in descending order
        sort_idx = np.argsort(-dets[:, -1])
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
            if dataset == 'imagenet':
                tp[j], fp[j] = _tpfp_imagenet(dets[j, :], gts[img_idx],
                                              covered[img_idx], iou_thr)
            else:
                tp[j], fp[j] = _tpfp_default(dets[j, :], gts[img_idx],
                                             covered[img_idx], iou_thr,
                                             gt_difficult[img_idx])
        # calculate precision and recall
        tp = np.cumsum(tp)
        fp = np.cumsum(fp)
        eps = np.finfo(np.float32).eps
        recall = tp / np.maximum(float(gt_num), eps)
        precision = tp / np.maximum((tp + fp), eps)
        # calculate AP
        mode = 'area' if dataset != 'voc07' else '11points'
        ap = average_precision(recall, precision, mode)
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
    mean_ap = np.array(aps).mean() if aps else 0.0
    if print_summary:
        print_map_summary(mean_ap, eval_results)
    return mean_ap, eval_results


def print_map_summary(mean_ap, results):
    """Print mAP and results of each class

    Args:
        mean_ap(float): calculated from `eval_map`
        results(list): calculated from `eval_map`
    """
    header = ['class', 'gts', 'dets', 'recall', 'precision', 'ap']
    table_data = [header]
    for i, cls_result in enumerate(results):

        recall = (cls_result['recall'][-1]
                  if cls_result['recall'].size > 0 else 0)
        precision = (cls_result['precision'][-1]
                     if cls_result['precision'].size > 0 else 0)
        row_data = [
            i + 1, cls_result['gt_num'], cls_result['det_num'],
            '{:.3f}'.format(recall), '{:.3f}'.format(precision),
            '{:.3f}'.format(cls_result['ap'])
        ]
        table_data.append(row_data)
    table_data.append(['mAP', '', '', '', '', '{:.3f}'.format(mean_ap)])
    table = AsciiTable(table_data)
    table.inner_footing_row_border = True
    print(table.table)
