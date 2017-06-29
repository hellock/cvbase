import pytest
from cvbase import list_cast, to_bool


def test_list_cast():
    assert list_cast([1, 2, 3], int) == [1, 2, 3]
    assert list_cast(['1.1', 2, '3'], float) == [1.1, 2.0, 3.0]
    assert list_cast([1, 2, 3], str) == ['1', '2', '3']
    with pytest.raises(TypeError):
        list_cast([1, 2, 3], '')
    with pytest.raises(TypeError):
        list_cast(1, str)


def test_to_bool():
    for var in ['false', 'False', '', False, 0]:
        assert to_bool(var) is False
    for var in ['true', 'True', True, 1]:
        assert to_bool(var) is True
