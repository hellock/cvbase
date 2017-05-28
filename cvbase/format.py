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
    """Convert a variable to bool type"""
    if isinstance(var, str):
        if var == '' or var.lower() == 'false':
            return False
        else:
            return True
    return bool(var)
