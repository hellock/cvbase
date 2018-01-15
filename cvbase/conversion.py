def list_cast(in_list, dst_type):
    """Convert a list of items to some type"""
    if not isinstance(in_list, list):
        raise TypeError('"in_list" must be a list')
    if not isinstance(dst_type, type):
        raise TypeError('"dst_type" must be a valid type')
    return list(map(dst_type, in_list))


def is_list_of(in_list, expected_type):
    """Check whether it is a list of objects of a certain type"""
    if not isinstance(in_list, list):
        return False
    for item in in_list:
        if not isinstance(item, expected_type):
            return False
    return True


def to_bool(var):
    """Convert a variable to bool type"""
    if isinstance(var, str):
        if var == '' or var.lower() == 'false':
            return False
        else:
            return True
    return bool(var)


def slice_list(in_list, lens):
    """Slice a list into several sub lists by a list of given length

    Args:
        in_list(list): the list to be sliced
        lens(int or list): the expected length of each out list

    Returns:
        list: list of sliced list
    """
    if not isinstance(lens, list):
        raise TypeError('"indices" must be a list of integers')
    elif sum(lens) != len(in_list):
        raise ValueError('list length and summation of lens do not match')
    out_list = []
    idx = 0
    for i in range(len(lens)):
        out_list.append(in_list[idx:idx + lens[i]])
        idx += lens[i]
    return out_list


def merge_list(in_list):
    """Merge a list of list into a single list

    Args:
        in_list(list): the list of list to be merged

    Returns:
        list: the flat list
    """
    out_list = []
    for sub_list in in_list:
        out_list.extend(sub_list)
    return out_list
