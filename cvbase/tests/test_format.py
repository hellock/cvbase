from cvbase.format import int_list, float_list, str_list


def test_int_list():
    in_list = ['1', '2', '3']
    assert int_list(in_list) == [1, 2, 3]


def test_float_list():
    in_list = ['1.1', 2, '3']
    assert float_list(in_list) == [1.1, 2.0, 3.0]


def test_str_list():
    in_list = [1, 2, 3]
    assert str_list(in_list) == ['1', '2', '3']
