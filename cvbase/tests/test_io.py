from os import makedirs, path, remove, removedirs

from cvbase.io import *


def test_json():
    tmp_filename = '.cvbase_test.tmp.json'
    test_obj = [{'a': 'abc', 'b': 1}, 2, 'c']
    json_dump(test_obj, tmp_filename)
    assert path.isfile(tmp_filename)
    load_obj = json_load(tmp_filename)
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


def test_read_img():
    img = read_img(path.join(path.dirname(__file__), 'data/test.jpg'))
    assert img.shape == (300, 400, 3)
