def int_list(in_list):
    """Convert a list of items to a list of int"""
    return list(map(int, in_list))


def float_list(in_list):
    """Convert a list of items to a list of float"""
    return list(map(float, in_list))


def str_list(in_list):
    """Convert a list of items to a list of str"""
    return list(map(str, in_list))


def to_bool(var):
    if isinstance(var, str):
        if str == '' or str.lower() == 'false':
            return False
        else:
            return True
    return bool(var)
