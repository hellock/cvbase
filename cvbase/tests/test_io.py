from os import path

from cvbase.io import read_img


def test_read_img():
    img = read_img(path.join(path.dirname(__file__), 'data/test.jpg'))
    assert img.shape == (300, 400, 3)
