import numpy as np


def read_flow(flow_or_path):
    """Read an optical flow map

    Args:
        flow_or_path(ndarray or str): either a flow map or path of a flow

    Returns:
        ndarray: optical flow
    """
    if isinstance(flow_or_path, np.ndarray):
        if (flow_or_path.ndim != 3) or (flow_or_path.shape[-1] != 2):
            raise ValueError(
                'Invalid flow with shape {}'.format(flow_or_path.shape))
        return flow_or_path
    elif not isinstance(flow_or_path, str):
        raise TypeError(
            '"flow_or_path" must be a filename or numpy array, not {}'.format(
                type(flow_or_path)))

    with open(flow_or_path, 'rb') as f:
        try:
            header = f.read(4)
            if header.decode('utf-8') != 'PIEH':
                raise IOError(
                    'Invalid flow file: {}, header does not contain PIEH'.
                    format(flow_or_path))
        except:
            raise IOError('Invalid flow file: {}'.format(flow_or_path))

        w = np.fromfile(f, np.int32, 1).squeeze()
        h = np.fromfile(f, np.int32, 1).squeeze()
        flow = np.fromfile(f, np.float32, w * h * 2).reshape((h, w, 2))

    return flow.astype(np.float32)


def write_flow(flow, filename):
    """Write optical flow to file

    Args:
        flow(ndarray): optical flow
        filename(str): file path
    """
    with open(filename, 'wb') as f:
        f.write('PIEH'.encode('utf-8'))
        np.array([flow.shape[1], flow.shape[0]], dtype=np.int32).tofile(f)
        flow = flow.astype(np.float32)
        flow.tofile(f)
        f.flush()
