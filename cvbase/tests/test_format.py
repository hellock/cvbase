from cvbase.format import int_list, float_list, str_list, to_bool


def test_int_list():
    in_list = ['1', '2', '3']
    assert int_list(in_list) == [1, 2, 3]


def test_float_list():
    in_list = ['1.1', 2, '3']
    assert float_list(in_list) == [1.1, 2.0, 3.0]


def test_str_list():
    in_list = [1, 2, 3]
    assert str_list(in_list) == ['1', '2', '3']


def test_to_bool():
    for var in ['false', 'False', '', False, 0]:
        assert to_bool(var) is False
    for var in ['true', 'True', True, 1]:
        assert to_bool(var) is True
