import sys
from os import makedirs, path, remove, removedirs

import cvbase as cvb
import pytest


def test_json():
    test_obj = [{'a': 'abc', 'b': 1}, 2, 'c']
    # json dump as a string
    json_str = cvb.json_dump(test_obj)
    assert json_str in [
        '[{"a": "abc", "b": 1}, 2, "c"]', '[{"b": 1, "a": "abc"}, 2, "c"]'
    ]
    # json load/dump with filename
    tmp_filename = '.cvbase_test.tmp.json'
    cvb.json_dump(test_obj, tmp_filename)
    assert path.isfile(tmp_filename)
    load_obj = cvb.json_load(tmp_filename)
    assert load_obj == test_obj
    remove(tmp_filename)
    # json load/dump with file-like object
    with open(tmp_filename, 'w') as f:
        cvb.json_dump(test_obj, f)
    assert path.isfile(tmp_filename)
    with open(tmp_filename, 'r') as f:
        load_obj = cvb.json_load(f)
    assert load_obj == test_obj
    remove(tmp_filename)


def test_yaml():
    test_obj = [{'a': 'abc', 'b': 1}, 2, 'c']
    # yaml dump as a string
    yaml_str = cvb.yaml_dump(test_obj)
    assert yaml_str in [
        '- {a: abc, b: 1}\n- 2\n- c\n', '- {b: 1, a: abc}\n- 2\n- c\n'
    ]
    # yaml load/dump with filename
    tmp_filename = '.cvbase_test.tmp.yaml'
    cvb.yaml_dump(test_obj, tmp_filename)
    assert path.isfile(tmp_filename)
    load_obj = cvb.yaml_load(tmp_filename)
    assert load_obj == test_obj
    remove(tmp_filename)
    # yaml load/dump with file-like object
    with open(tmp_filename, 'w') as f:
        cvb.yaml_dump(test_obj, f)
    assert path.isfile(tmp_filename)
    with open(tmp_filename, 'r') as f:
        load_obj = cvb.yaml_load(f)
    assert load_obj == test_obj
    remove(tmp_filename)


def test_pickle():
    test_obj = [{'a': 'abc', 'b': 1}, 2, 'c']
    # pickle dump as a string
    pickle_str = cvb.pickle_dump(test_obj)
    import pickle
    assert pickle.loads(pickle_str) == test_obj
    # pickle load/dump with filename
    tmp_filename = '.cvbase_test.tmp.pkl'
    cvb.pickle_dump(test_obj, tmp_filename)
    assert path.isfile(tmp_filename)
    load_obj = cvb.pickle_load(tmp_filename)
    assert load_obj == test_obj
    remove(tmp_filename)
    # pickle load/dump with file-like object
    with open(tmp_filename, 'wb') as f:
        cvb.pickle_dump(test_obj, f)
    assert path.isfile(tmp_filename)
    with open(tmp_filename, 'rb') as f:
        load_obj = cvb.pickle_load(f)
    assert load_obj == test_obj
    remove(tmp_filename)


def test_universal():
    test_obj = [{'a': 'abc', 'b': 1}, 2, 'c']
    # dump as a string
    for format in ['json', 'yaml', 'pickle']:
        cvb.dump(test_obj, format=format)
    with pytest.raises(ValueError):
        cvb.dump(test_obj)
    with pytest.raises(TypeError):
        cvb.dump(test_obj, 'tmp.txt')
    # test load/dump with filename
    for format in ['json', 'yaml', 'pkl']:
        tmp_filename = '.cvbase_test.tmp.' + format
        cvb.dump(test_obj, tmp_filename)
        assert path.isfile(tmp_filename)
        load_obj = cvb.load(tmp_filename)
        assert load_obj == test_obj
        remove(tmp_filename)
    # test json load/dump with file object
    for format in ['json', 'yaml', 'pkl']:
        tmp_filename = '.cvbase_test.tmp.' + format
        mode = 'wb' if format == 'pkl' else 'w'
        with open(tmp_filename, mode) as f:
            cvb.dump(test_obj, f, format=format)
        assert path.isfile(tmp_filename)
        mode = 'rb' if format == 'pkl' else 'r'
        with open(tmp_filename, mode) as f:
            load_obj = cvb.load(f, format=format)
        assert load_obj == test_obj
        remove(tmp_filename)


def test_list_from_file():
    filename = path.join(path.dirname(__file__), 'data/filelist.txt')
    filelist = cvb.list_from_file(filename)
    assert filelist == ['1.jpg', '2.jpg', '3.jpg', '4.jpg', '5.jpg']
    filelist = cvb.list_from_file(filename, prefix='a/')
    assert filelist == ['a/1.jpg', 'a/2.jpg', 'a/3.jpg', 'a/4.jpg', 'a/5.jpg']
    filelist = cvb.list_from_file(filename, offset=2)
    assert filelist == ['3.jpg', '4.jpg', '5.jpg']
    filelist = cvb.list_from_file(filename, max_num=2)
    assert filelist == ['1.jpg', '2.jpg']
    filelist = cvb.list_from_file(filename, offset=3, max_num=3)
    assert filelist == ['4.jpg', '5.jpg']


def test_dict_from_file():
    filename = path.join(path.dirname(__file__), 'data/kv_pair.txt')
    mapping = cvb.dict_from_file(filename)
    assert mapping == {'1': 'cat', '2': ['dog', 'cow'], '3': 'panda'}
    mapping = cvb.dict_from_file(filename, key_type=int)
    assert mapping == {1: 'cat', 2: ['dog', 'cow'], 3: 'panda'}


def test_async_dumper():
    tmp_dir = '.cvbase_test.tmp'
    makedirs(tmp_dir)
    dump_proc = cvb.AsyncDumper()
    dump_proc.start()
    for i in range(3):
        dump_proc.dump(i, path.join(tmp_dir, '{}.pkl'.format(i)))
    dump_proc.dump(None, None)
    dump_proc.join()
    assert not dump_proc.is_alive()
    for i in range(3):
        filename = path.join(tmp_dir, '{}.pkl'.format(i))
        assert path.isfile(filename)
        remove(filename)
    removedirs(tmp_dir)


def test_check_file_exist():
    cvb.check_file_exist(__file__)
    if sys.version_info > (3, 3):
        with pytest.raises(FileNotFoundError):
            cvb.check_file_exist('not_a_file.txt')
    else:
        with pytest.raises(IOError):
            cvb.check_file_exist('not_a_file.txt')


def test_scandir():
    folder = path.join(path.dirname(__file__), 'data')
    assert set(cvb.scandir(folder)) == set([
        'filelist.txt', 'kv_pair.txt', 'voc_labels.txt', 'color.jpg',
        'grayscale.jpg', 'test.mp4'
    ])
    assert set(cvb.scandir(folder, '.txt')) == set(
        ['filelist.txt', 'kv_pair.txt', 'voc_labels.txt'])
    assert set(cvb.scandir(folder, ('.jpg', '.txt'))) == set([
        'filelist.txt', 'kv_pair.txt', 'voc_labels.txt', 'color.jpg',
        'grayscale.jpg'
    ])
    with pytest.raises(TypeError):
        cvb.scandir(folder, 111)
