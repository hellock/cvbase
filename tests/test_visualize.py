from os import path

import cvbase as cvb


def test_read_labels():
    label_names = ['a', 'b', 'c']
    assert cvb.read_labels(label_names) == label_names
    voc_labels = [
        'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat',
        'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person',
        'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'
    ]
    assert cvb.read_labels('voc') == voc_labels
    label_file = path.join(path.dirname(__file__), 'data/voc_labels.txt')
    assert cvb.read_labels(label_file) == voc_labels
