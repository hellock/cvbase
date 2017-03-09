import json
try:
    import cPickle as pickle
except:
    import pickle
from multiprocessing import Process, Queue


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
