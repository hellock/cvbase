import numpy as np

from cvbase.image import read_img, write_img
from cvbase.opencv import IMREAD_UNCHANGED


def _pair_name(filename):
    parts = filename.split('.')
    parts[-2] += '_dx'
    dx_filename = '.'.join(parts)
    parts[-2] = parts[-2][:-1] + 'y'
    dy_filename = '.'.join(parts)
    return dx_filename, dy_filename


def read_flow(flow_or_path, quantize=False, *args, **kwargs):
    """Read an optical flow map

    Args:
        flow_or_path(ndarray or str): either a flow map or path of a flow
        quantize(bool): whether to read quantized pair, if set to True,
                        remaining args will be passed to :func:`dequantize_flow`

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

    if not quantize:
        with open(flow_or_path, 'rb') as f:
            try:
                header = f.read(4).decode('utf-8')
            except:
                raise IOError('Invalid flow file: {}'.format(flow_or_path))
            else:
                if header != 'PIEH':
                    raise IOError(
                        'Invalid flow file: {}, header does not contain PIEH'.
                        format(flow_or_path))

            w = np.fromfile(f, np.int32, 1).squeeze()
            h = np.fromfile(f, np.int32, 1).squeeze()
            flow = np.fromfile(f, np.float32, w * h * 2).reshape((h, w, 2))
    else:
        dx_filename, dy_filename = _pair_name(flow_or_path)
        dx = read_img(dx_filename, flag=IMREAD_UNCHANGED)
        dy = read_img(dy_filename, flag=IMREAD_UNCHANGED)
        flow = dequantize_flow(dx, dy, *args, **kwargs)

    return flow.astype(np.float32)


def write_flow(flow, filename, quantize=False, *args, **kwargs):
    """Write optical flow to file

    Args:
        flow(ndarray): optical flow
        filename(str): file path
        quantize(bool): whether to quantize the flow and save as 2 images,
                        if set to True, remaining args will be passed
                        to :func:`quantize_flow`
    """
    if not quantize:
        with open(filename, 'wb') as f:
            f.write('PIEH'.encode('utf-8'))
            np.array([flow.shape[1], flow.shape[0]], dtype=np.int32).tofile(f)
            flow = flow.astype(np.float32)
            flow.tofile(f)
            f.flush()
    else:
        dx, dy = quantize_flow(flow, *args, **kwargs)
        dx_filename, dy_filename = _pair_name(filename)
        write_img(dx, dx_filename)
        write_img(dy, dy_filename)


def quantize_flow(flow, max_val=0.02, norm=True):
    """Quantize flow to [0, 255] (much smaller size when dumping as images)

    Args:
        flow(ndarray): optical flow
        max_val(float): maximum value of flow, values beyond
                        [-max_val, max_val] will be truncated.
        norm(bool): whether to divide flow values by width/height

    Returns:
        tuple: quantized dx and dy
    """
    h, w, _ = flow.shape
    dx = flow[..., 0]
    dy = flow[..., 1]
    if norm:
        dx = dx / w  # avoid inplace operations
        dy = dy / h
    dx = np.maximum(0, np.minimum(dx + max_val, 2 * max_val))
    dy = np.maximum(0, np.minimum(dy + max_val, 2 * max_val))
    dx = np.round(dx * 255 / (max_val * 2)).astype(np.uint8)
    dy = np.round(dy * 255 / (max_val * 2)).astype(np.uint8)
    return dx, dy


def dequantize_flow(dx, dy, max_val=0.02, denorm=True):
    """Recover flow from quantized flow

    Args:
        dx(ndarray): quantized dx
        dy(ndarray): quantized dy
        max_val(float): maximum value used when quantizing.
        denorm(bool): whether to multiply flow values with width/height

    Returns:
        tuple: dequantized dx and dy
    """
    assert dx.shape == dy.shape
    assert dx.ndim == 2 or (dx.ndim == 3 and dx.shape[-1] == 1)
    dx = dx.astype(np.float32) * max_val * 2 / 255 - max_val
    dy = dy.astype(np.float32) * max_val * 2 / 255 - max_val
    if denorm:
        dx *= dx.shape[1]
        dy *= dx.shape[0]
    flow = np.dstack((dx, dy))
    return flow
