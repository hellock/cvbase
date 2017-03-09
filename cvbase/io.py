import json
try:
    import cPickle as pickle
except:
    import pickle
from multiprocessing import Process, Queue
from os import path

import cv2
import numpy as np


def json_dump(obj, filename, **kwargs):
    with open(filename, 'wb') as f:
        json.dump(obj, f, **kwargs)


def json_load(filename):
    with open(filename, 'rb') as f:
        obj = json.load(f)
    return obj


def pickle_dump(obj, filename, protocol=2, **kwargs):
    with open(filename, 'wb') as f:
        pickle.dump(obj, f, protocol=protocol, **kwargs)


def pickle_load(filename):
    with open(filename, 'rb') as f:
        obj = pickle.load(f)
    return obj


class AsyncDumper(Process):
    def __init__(self):
        self._io_queue = Queue()
        super(AsyncDumper, self).__init__()

    def run(self):
        while True:
            data, out_file = self._io_queue.get()
            if data is None:
                break
            pickle_dump(data, out_file)

    def dump(self, obj, filename):
        self._io_queue.put((obj, filename))


def read_img(img):
    if isinstance(img, np.ndarray):
        return img
    elif isinstance(img, str):
        if path.isfile(img):
            return cv2.imread(img)
        else:
            raise IOError('img file does not exist: {}'.format(img))
    else:
        raise TypeError('"img" must be a numpy array or a filename')
