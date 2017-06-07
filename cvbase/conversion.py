def list_cast(in_list, dst_type):
    """Convert a list of items to some type"""
    if not isinstance(in_list, list):
        raise TypeError('"in_list" must be a list')
    if not isinstance(dst_type, type):
        raise TypeError('"dst_type" must be a valid type')
    return list(map(dst_type, in_list))


def int_list(in_list):
    """(deprecated)Convert a list of items to a list of int"""
    return list(map(int, in_list))


def float_list(in_list):
    """(deprecated)Convert a list of items to a list of float"""
    return list(map(float, in_list))


def str_list(in_list):
    """(deprecated)Convert a list of items to a list of str"""
    return list(map(str, in_list))


def to_bool(var):
    """Convert a variable to bool type"""
    if isinstance(var, str):
        if var == '' or var.lower() == 'false':
            return False
        else:
            return True
    return bool(var)
