import numpy as np


def bbox_transform(proposals, gt):
    """Calculate regression deltas from proposals and ground truths

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
    return deltas


def bbox_transform_inv(bboxes, deltas):
    """Get ground truth bboxes from input bboxes and deltas

    gw = pw * exp(dw), gx = px + dx * pw

    Args:
        bboxes(ndarray): shape (..., 4) [x1, y1, x2, y2]
        deltas(ndarray): shape (..., 4*k) [dx, dy, dw, dh]
    Output:
        ndarray: same shape as input deltas
    """

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
    bboxes[..., 0::4] = np.maximum(
        np.minimum(bboxes[..., 0::4], img_shape[1] - 1), 0)
    bboxes[..., 1::4] = np.maximum(
        np.minimum(bboxes[..., 1::4], img_shape[0] - 1), 0)
    bboxes[..., 2::4] = np.maximum(
        np.minimum(bboxes[..., 2::4], img_shape[1] - 1), 0)
    bboxes[..., 3::4] = np.maximum(
        np.minimum(bboxes[..., 3::4], img_shape[0] - 1), 0)
    return bboxes


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


def bbox_normalize(deltas, means=[0, 0, 0, 0], stds=[1, 1, 1, 1]):
    """Normalize bbox deltas

    Args:
        deltas(ndarray): shape(..., 4*k)
        means(ndarray or list): shape(4, ) or (4*k, )
        stds(ndarray or list): shape(4, ) or (4*k, )
    Output:
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
    Output:
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
