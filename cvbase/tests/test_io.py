from os import makedirs, path, remove, removedirs

from cvbase.io import (json_dump, json_load, yaml_load, yaml_dump, pickle_dump,
                       pickle_load, list_from_file, AsyncDumper, scandir)


def test_json():
    tmp_filename = '.cvbase_test.tmp.json'
    test_obj = [{'a': 'abc', 'b': 1}, 2, 'c']
    json_str = json_dump(test_obj)
    assert json_str in [
        '[{"a": "abc", "b": 1}, 2, "c"]', '[{"b": 1, "a": "abc"}, 2, "c"]'
    ]
    json_dump(test_obj, tmp_filename)
    assert path.isfile(tmp_filename)
    load_obj = json_load(tmp_filename)
    assert load_obj == test_obj
    remove(tmp_filename)


def test_yaml():
    tmp_filename = '.cvbase_test.tmp.yaml'
    test_obj = [{'a': 'abc', 'b': 1}, 2, 'c']
    yaml_str = yaml_dump(test_obj)
    assert yaml_str in [
        '- {a: abc, b: 1}\n- 2\n- c\n', '- {b: 1, a: abc}\n- 2\n- c\n'
    ]
    yaml_dump(test_obj, tmp_filename)
    assert path.isfile(tmp_filename)
    load_obj = yaml_load(tmp_filename)
    assert load_obj == test_obj
    remove(tmp_filename)


def test_pickle():
    tmp_filename = '.cvbase_test.tmp.pkl'
    test_obj = [{'a': 'abc', 'b': 1}, 2, 'c']
    pickle_dump(test_obj, tmp_filename)
    assert path.isfile(tmp_filename)
    load_obj = pickle_load(tmp_filename)
    assert load_obj == test_obj
    remove(tmp_filename)


def test_list_from_file():
    filename = path.join(path.dirname(__file__), 'data/filelist.txt')
    filelist = list_from_file(filename)
    assert filelist == ['1.jpg', '2.jpg', '3.jpg', '4.jpg', '5.jpg']
    filelist = list_from_file(filename, prefix='a/')
    assert filelist == ['a/1.jpg', 'a/2.jpg', 'a/3.jpg', 'a/4.jpg', 'a/5.jpg']
    filelist = list_from_file(filename, offset=2)
    assert filelist == ['3.jpg', '4.jpg', '5.jpg']
    filelist = list_from_file(filename, max_num=2)
    assert filelist == ['1.jpg', '2.jpg']
    filelist = list_from_file(filename, offset=3, max_num=3)
    assert filelist == ['4.jpg', '5.jpg']


def test_async_dumper():
    tmp_dir = '.cvbase_test.tmp'
    makedirs(tmp_dir)
    dump_proc = AsyncDumper()
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


def test_scandir():
    folder = path.join(path.dirname(__file__), 'data')
    assert set(scandir(folder)) == set(
        ['filelist.txt', 'test.jpg', 'voc_labels.txt'])
    assert set(scandir(folder, '.txt')) == set(
        ['filelist.txt', 'voc_labels.txt'])
    assert set(scandir(folder, ('.jpg', '.txt'))) == set(
        ['filelist.txt', 'test.jpg', 'voc_labels.txt'])
