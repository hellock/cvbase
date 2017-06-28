import json
import os
import sys
try:
    import cPickle as pickle
except:
    import pickle
import yaml
try:
    from yaml import CLoader as Loader, CDumper as Dumper
except ImportError:
    from yaml import Loader, Dumper
from multiprocessing import Process, Queue
from os import path


def json_load(filename):
    with open(filename, 'r') as f:
        obj = json.load(f)
    return obj


def json_dump(obj, filename=None, **kwargs):
    if filename is None:
        return json.dumps(obj, **kwargs)
    with open(filename, 'w') as f:
        json.dump(obj, f, **kwargs)


def yaml_load(filename, **kwargs):
    kwargs.setdefault('Loader', Loader)
    with open(filename, 'r') as f:
        obj = yaml.load(f, **kwargs)
    return obj


def yaml_dump(obj, filename=None, **kwargs):
    kwargs.setdefault('Dumper', Dumper)
    if filename is None:
        return yaml.dump(obj, **kwargs)
    with open(filename, 'w') as f:
        yaml.dump(obj, f, **kwargs)


def pickle_load(filename, **kwargs):
    with open(filename, 'rb') as f:
        obj = pickle.load(f, **kwargs)
    return obj


def pickle_dump(obj, filename, **kwargs):
    kwargs.setdefault('protocol', 2)
    with open(filename, 'wb') as f:
        pickle.dump(obj, f, **kwargs)


def list_from_file(filename, prefix='', offset=0, max_num=0):
    cnt = 0
    item_list = []
    with open(filename, 'r') as f:
        for _ in range(offset):
            f.readline()
        for line in f:
            if max_num > 0 and cnt >= max_num:
                break
            item_list.append(prefix + line.rstrip('\n'))
            cnt += 1
    return item_list


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


def check_file_exist(filename, msg):
    if not path.isfile(filename):
        try:
            raise FileNotFoundError(msg)
        except:
            raise IOError(msg)


def mkdir_or_exist(dir_name):
    if not path.isdir(dir_name):
        os.makedirs(dir_name)


def _scandir_py35(dir_path, suffix=None):
    for entry in os.scandir(dir_path):
        if not entry.is_file():
            continue
        filename = entry.name
        if suffix is None:
            yield filename
        elif filename.endswith(suffix):
            yield filename


def _scandir_py(dir_path, suffix=None):
    for filename in os.listdir(dir_path):
        if not path.isfile(path.join(dir_path, filename)):
            continue
        if suffix is None:
            yield filename
        elif filename.endswith(suffix):
            yield filename


def scandir(dir_path, suffix=None):
    if suffix is not None and not isinstance(suffix, (str, tuple)):
        raise TypeError('"suffix" must be a string or tuple of strings')
    if sys.version[0] == 3 and sys.version[1] >= 5:
        return _scandir_py35(dir_path, suffix)
    else:
        return _scandir_py(dir_path, suffix)
