import cv2
import numpy as np

from cvbase.io import read_img


def draw_bboxes(img, bboxes, colors=(0, 255, 0), top_k=0, thickness=1,
                show=True, win_name='', wait_time=0, out_file=None):  # yapf: disable
    img = read_img(img)
    if isinstance(bboxes, np.ndarray):
        bboxes = [bboxes]
    if isinstance(colors, tuple):
        colors = [colors for _ in range(len(bboxes))]
    assert len(bboxes) == len(colors)

    for i, _bboxes in enumerate(bboxes):
        if top_k <= 0:
            _top_k = _bboxes.shape[0]
        else:
            _top_k = min(top_k, _bboxes.shape[0])
        for j in range(_top_k):
            cv2.rectangle(img, (_bboxes[j, 0], _bboxes[j, 1]),
                          (_bboxes[j, 2], _bboxes[j, 3]), colors[i])
    if show:
        cv2.imshow(win_name, img)
        cv2.waitKey(wait_time)
    if out_file is not None:
        cv2.imwrite(out_file, img)
