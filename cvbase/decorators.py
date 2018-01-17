from functools import wraps
from importlib import import_module


def requires_package(package):

    def wrap(func):

        @wraps(func)
        def wrapped_func(*args, **kwargs):
            required_packages = [package] if isinstance(package,
                                                        str) else package
            missing = []
            for p in required_packages:
                try:
                    import_module(p)
                except:
                    missing.append(p)
            if missing:
                print('Package "{}" is required in method "{}" but not found'
                      ', please install the missing packages first.'.format(
                          ', '.join(missing), func.__name__))
                raise ImportError
            else:
                return func(*args, **kwargs)

        return wrapped_func

    return wrap
