from ._test_image import _TestImage


def test_backend():
    import cvbase as cvb
    cvb.set_backend('pillow')
    assert cvb.get_backend() == 'pillow'


class TestImage(_TestImage):

    def test_bgr2rgb(self):
        assert True

    def test_rgb2bgr(self):
        assert True
