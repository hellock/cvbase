import cvbase as cvb


def test_requires_package(capsys):

    @cvb.requires_package('nnn')
    def func_a():
        pass

    @cvb.requires_package(['n1', 'n2'])
    def func_b():
        pass

    func_a()
    out, _ = capsys.readouterr()
    assert out == ('Package "nnn" is required in method "func_a" but '
                   'not found, please install the missing packages first.\n')

    func_b()
    out, _ = capsys.readouterr()
    assert out == ('Package "n1, n2" is required in method "func_b" but '
                   'not found, please install the missing packages first.\n')
