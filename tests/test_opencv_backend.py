from ._test_image import _TestImage


def test_backend():
    import cvbase as cvb
    cvb.set_backend('opencv')
    assert cvb.get_backend() == 'opencv'


class TestImage(_TestImage):

    def test_new(self):
        super(TestImage, self).test_new()
        import cvbase.image as im
        import numpy as np
        img = im.Image.new('RGB', (200, 100), 0.1)
        assert (img.numpy().shape == (100, 200, 3)
                and img.numpy().dtype == np.float32)