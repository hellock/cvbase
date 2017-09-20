import numpy as np


def bbox_overlaps(bboxes1, bboxes2):
    """Calculate the ious between each bbox of bboxes1 and bboxes2

    Args:
        bboxes1(ndarray): shape (n, 4)
        bboxes2(ndarray): shape (k, 4)

    Returns:
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


def bbox2roi(bbox_list, stack=True):
    """Convert bboxes to rois by adding index at the first col.

    Args:
        bbox_list(list): a list of ndarray (k_i, 4)
        stack(bool): whether to stack all the rois

    Returns:
        ndarray or list: rois of shape (sum_k, 4)
    """
    if not bbox_list:
        return np.zeros((0, 5), dtype=np.float32)
    batch_rois = []
    for i, bboxes in enumerate(bbox_list):
        num_rois = bboxes.shape[0]
        batch_inds = np.empty((num_rois, 1), dtype=np.float32)
        batch_inds.fill(float(i))
        batch_rois.append(np.hstack([batch_inds, bboxes]))
    if stack:
        batch_rois = np.vstack(batch_rois)
    return batch_rois


def bbox_normalize(deltas, means=[0, 0, 0, 0], stds=[1, 1, 1, 1]):
    """Normalize bbox deltas

    Args:
        deltas(ndarray): shape(..., 4*k)
        means(ndarray or list): shape(4, ) or (4*k, )
        stds(ndarray or list): shape(4, ) or (4*k, )

    Returns:
        ndarray: normalized deltas, same shape as input deltas
    """
    if isinstance(means, list):
        means = np.array(means)
    if isinstance(stds, list):
        stds = np.array(stds)
    assert deltas.shape[-1] % 4 == 0
    assert means.size == 4 or means.size == deltas.shape[-1]
    assert stds.shape == means.shape

    if means.size == 4 and deltas.shape[-1] > 4:
        reps = list(deltas.shape)
        reps[-1] = reps[-1] // 4
        means = np.tile(means, tuple(reps))
        stds = np.tile(stds, tuple(reps))
    return (deltas - means) / stds


def bbox_denormalize(deltas, means=[0, 0, 0, 0], stds=[1, 1, 1, 1]):
    """Denormalize bbox deltas

    Args:
        deltas(ndarray): shape(..., 4*k)
        means(ndarray or list): shape(4, ) or (4*k, )
        stds(ndarray or list): shape(4, ) or (4*k, )

    Returns:
        ndarray: denormalized deltas, same shape as input deltas
    """
    if isinstance(means, list):
        means = np.array(means)
    if isinstance(stds, list):
        stds = np.array(stds)
    assert deltas.shape[-1] % 4 == 0
    assert means.size == 4 or means.size == deltas.shape[-1]
    assert stds.shape == means.shape

    if means.size == 4 and deltas.shape[-1] > 4:
        reps = list(deltas.shape)
        reps[-1] = reps[-1] // 4
        means = np.tile(means, tuple(reps))
        stds = np.tile(stds, tuple(reps))
    return deltas * stds + means


def bbox_transform(proposals, gt, means=[0, 0, 0, 0], stds=[1, 1, 1, 1]):
    """Calculate regression deltas from proposals and ground truths

    dx = (gx - px) / pw, dw = log(gw / pw)

    Args:
        proposals(ndarray): shape (..., 4)
        gt(ndarray): shape (..., 4) or (1.., 4)

    Returns:
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
    px = (proposals[..., 0] + proposals[..., 2]) * 0.5
    py = (proposals[..., 1] + proposals[..., 3]) * 0.5
    pw = proposals[..., 2] - proposals[..., 0] + 1.0
    ph = proposals[..., 3] - proposals[..., 1] + 1.0

    gx = (gt[..., 0] + gt[..., 2]) * 0.5
    gy = (gt[..., 1] + gt[..., 3]) * 0.5
    gw = gt[..., 2] - gt[..., 0] + 1.0
    gh = gt[..., 3] - gt[..., 1] + 1.0

    dx = (gx - px) / pw
    dy = (gy - py) / ph
    dw = np.log(gw / pw)
    dh = np.log(gh / ph)
    deltas = np.concatenate(
        (dx[..., np.newaxis], dy[..., np.newaxis], dw[..., np.newaxis],
         dh[..., np.newaxis]),
        axis=-1)
    if means != [0, 0, 0, 0] or stds != [1, 1, 1, 1]:
        deltas = bbox_normalize(deltas, means, stds)
    return deltas


def bbox_transform_inv(bboxes, deltas, means=[0, 0, 0, 0], stds=[1, 1, 1, 1]):
    """Get ground truth bboxes from input bboxes and deltas

    gw = pw * exp(dw), gx = px + dx * pw

    Args:
        bboxes(ndarray): shape (..., 4) [x1, y1, x2, y2]
        deltas(ndarray): shape (..., 4*k) [dx, dy, dw, dh]

    Returns:
        ndarray: same shape as input deltas
    """
    if means != [0, 0, 0, 0] or stds != [1, 1, 1, 1]:
        deltas = bbox_denormalize(deltas, means, stds)
    px = (bboxes[..., 0] + bboxes[..., 2]) * 0.5
    py = (bboxes[..., 1] + bboxes[..., 3]) * 0.5
    pw = bboxes[..., 2] - bboxes[..., 0] + 1.0
    ph = bboxes[..., 3] - bboxes[..., 1] + 1.0
    gw = pw[..., np.newaxis] * np.exp(deltas[..., 2::4])
    gh = ph[..., np.newaxis] * np.exp(deltas[..., 3::4])
    gx = px[..., np.newaxis] + pw[..., np.newaxis] * deltas[..., 0::4]
    gy = py[..., np.newaxis] + ph[..., np.newaxis] * deltas[..., 1::4]
    shape = list(gx.shape)
    shape[-1] = shape[-1] * 4
    return np.stack(
        (gx - gw * 0.5 + 0.5, gy - gh * 0.5 + 0.5, gx + gw * 0.5 - 0.5,
         gy + gh * 0.5 - 0.5),
        axis=-1).reshape(tuple(shape))


def bbox_clip(bboxes, img_shape):
    """Limit bboxes to fit the image size

    Args:
        bboxes(ndarray): shape (..., 4*k)
        img_shape(tuple): (height, width)
    """
    assert bboxes.shape[-1] % 4 == 0
    cliped_bboxes = np.empty_like(bboxes, dtype=bboxes.dtype)
    cliped_bboxes[..., 0::4] = np.maximum(
        np.minimum(bboxes[..., 0::4], img_shape[1] - 1), 0)
    cliped_bboxes[..., 1::4] = np.maximum(
        np.minimum(bboxes[..., 1::4], img_shape[0] - 1), 0)
    cliped_bboxes[..., 2::4] = np.maximum(
        np.minimum(bboxes[..., 2::4], img_shape[1] - 1), 0)
    cliped_bboxes[..., 3::4] = np.maximum(
        np.minimum(bboxes[..., 3::4], img_shape[0] - 1), 0)
    return cliped_bboxes


def bbox_flip(bboxes, img_shape):
    """Flip bboxes horizontally

    Args:
        bboxes(ndarray): shape (..., 4*k)
        img_shape(tuple): (height, width)
    """
    assert bboxes.shape[-1] % 4 == 0
    w = img_shape[1]
    flipped = bboxes.copy()
    flipped[..., 0::4] = w - bboxes[..., 2::4] - 1
    flipped[..., 2::4] = w - bboxes[..., 0::4] - 1
    return flipped


def bbox_scaling(bboxes, scale, clip_shape=None):
    """Scaling bboxes and clip the boundary(optional)

    Args:
        bboxes(ndarray): shape(..., 4)
        scale(float): scaling factor
        clip(None or tuple): (h, w)

    Returns:
        ndarray: scaled bboxes
    """
    if float(scale) == 1.0:
        scaled_bboxes = bboxes.copy()
    else:
        w = bboxes[..., 2] - bboxes[..., 0] + 1
        h = bboxes[..., 3] - bboxes[..., 1] + 1
        dw = (w * (scale - 1)) * 0.5
        dh = (h * (scale - 1)) * 0.5
        scaled_bboxes = bboxes + np.stack((-dw, -dh, dw, dh), axis=-1)
    if clip_shape is not None:
        return bbox_clip(scaled_bboxes, clip_shape)
    else:
        return scaled_bboxes


def bbox_perturb(bbox,
                 offset_ratio,
                 num,
                 clip_shape=None,
                 min_iou=None,
                 max_iou=None,
                 max_try=20):
    """Perturb a bbox around it to generate more bboxes

    Args:
        bbox(ndarray): shape(4,)
        offset_ratio(float): max offset ratio (w.r.t the bbox w and h)
        num(int): number of bboxes to be generated
        clip_shape(None or tuple): (h, w)
        min_iou(float): minimum iou of perturbed bboxes with original bbox
        max_iou(float): maximum iou of perturbed bboxes with original bbox

    Returns:
        ndarray: perturbed bboxes of shape (num, 4)
    """
    w = bbox[2] - bbox[0] + 1
    h = bbox[3] - bbox[1] + 1
    max_offset = np.array([w, h, w, h], dtype=np.float32) * offset_ratio
    # generate more bboxes to satisfy the iou condition more easily
    num_relaxed = num * 2 if min_iou or max_iou else num
    p_bboxes = np.zeros((num_relaxed, 4), dtype=np.float32)
    for i in range(4):
        p_bboxes[:, i] = np.random.uniform(
            bbox[i] - max_offset[i], bbox[i] + max_offset[i], num_relaxed)
    if clip_shape:
        p_bboxes = bbox_clip(p_bboxes, clip_shape)
    inds_valid = (p_bboxes[:, 0] < p_bboxes[:, 2]) & (p_bboxes[:, 1] <
                                                      p_bboxes[:, 3])
    p_bboxes = p_bboxes[inds_valid, :]
    if min_iou or max_iou:
        min_iou = 0 if min_iou is None else min_iou
        max_iou = 2 if max_iou is None else max_iou
        ious = bbox_overlaps(bbox[np.newaxis, ...], p_bboxes)[0, :]
        inds_keep = (ious >= min_iou) & (ious < max_iou)
        p_bboxes = p_bboxes[inds_keep, ...]
    if p_bboxes.shape[0] < num:
        if max_try > 1:
            extra_bboxes = bbox_perturb(bbox, offset_ratio,
                                        num - p_bboxes.shape[0], clip_shape,
                                        min_iou, max_iou, max_try - 1)
            return np.vstack((p_bboxes, extra_bboxes))
        else:
            return p_bboxes
    elif p_bboxes.shape[0] > num:
        return p_bboxes[:num, :]
    else:
        return p_bboxes
