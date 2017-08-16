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


def json_load(file):
    if isinstance(file, str):
        with open(file, 'r') as f:
            obj = json.load(f)
    elif hasattr(file, 'read'):
        obj = json.load(file)
    else:
        raise TypeError('"file" must be a filename str or a file-object')
    return obj


def json_dump(obj, file=None, **kwargs):
    if file is None:
        return json.dumps(obj, **kwargs)
    elif isinstance(file, str):
        with open(file, 'w') as f:
            json.dump(obj, f, **kwargs)
    elif hasattr(file, 'write'):
        json.dump(obj, file, **kwargs)
    else:
        raise TypeError('"file" must be a filename str or a file-object')


def yaml_load(file, **kwargs):
    kwargs.setdefault('Loader', Loader)
    if isinstance(file, str):
        with open(file, 'r') as f:
            obj = yaml.load(f, **kwargs)
    elif hasattr(file, 'read'):
        obj = yaml.load(file, **kwargs)
    else:
        raise TypeError('"file" must be a filename str or a file-object')
    return obj


def yaml_dump(obj, file=None, **kwargs):
    kwargs.setdefault('Dumper', Dumper)
    if file is None:
        return yaml.dump(obj, **kwargs)
    elif isinstance(file, str):
        with open(file, 'w') as f:
            yaml.dump(obj, f, **kwargs)
    elif hasattr(file, 'write'):
        yaml.dump(obj, file, **kwargs)
    else:
        raise TypeError('"file" must be a filename str or a file-object')


def pickle_load(file, **kwargs):
    if isinstance(file, str):
        with open(file, 'rb') as f:
            obj = pickle.load(f, **kwargs)
    elif hasattr(file, 'read'):
        obj = pickle.load(file, **kwargs)
    else:
        raise TypeError('"file" must be a filename str or a file-object')
    return obj


def pickle_dump(obj, file=None, **kwargs):
    kwargs.setdefault('protocol', 2)
    if file is None:
        return pickle.dumps(obj, **kwargs)
    elif isinstance(file, str):
        with open(file, 'wb') as f:
            pickle.dump(obj, f, **kwargs)
    elif hasattr(file, 'write'):
        pickle.dump(obj, file, **kwargs)
    else:
        raise TypeError('"file" must be a filename str or a file-object')


def load(file, format=None, **kwargs):
    processors = {
        'json': json_load,
        'yaml': yaml_load,
        'yml': yaml_load,
        'pickle': pickle_load,
        'pkl': pickle_load
    }
    if format is None and isinstance(file, str):
        format = file.split('.')[-1]
    if format not in processors:
        raise TypeError('Unsupported format: ' + format)
    return processors[format](file, **kwargs)


def dump(obj, file=None, format=None, **kwargs):
    processors = {
        'json': json_dump,
        'yaml': yaml_dump,
        'yml': yaml_dump,
        'pickle': pickle_dump,
        'pkl': pickle_dump
    }
    if format is None:
        if isinstance(file, str):
            format = file.split('.')[-1]
        elif file is None:
            raise ValueError('format must be specified')
    if format not in processors:
        raise TypeError('Unsupported format: ' + format)
    return processors[format](obj, file, **kwargs)


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


def dict_from_file(filename, key_type=str):
    mapping = {}
    with open(filename, 'r') as f:
        for line in f:
            items = line.rstrip('\n').split(' ')
            assert len(items) >= 2
            key = key_type(items[0])
            val = items[1:] if len(items) > 2 else items[1]
            mapping[key] = val
    return mapping


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


def check_file_exist(filename, msg_tmpl='file "{}" not exist:'):
    if not path.isfile(filename):
        if sys.version_info > (3, 3):
            raise FileNotFoundError(msg_tmpl.format(filename))
        else:
            raise IOError(msg_tmpl.format(filename))


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
