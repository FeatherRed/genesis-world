import numpy as np

from genesis.utils.device_buffer import DeviceView
from genesis.utils.torch_compat import torch


def brief(x):
    if hasattr(x, "_repr_brief"):
        return x._repr_brief()

    elif isinstance(x, (int, float, dict, bool, list, tuple, np.integer, np.floating)):
        return f"{__repr_name__(x)}: {x}"

    elif isinstance(x, str):
        return f"{__repr_name__(x)}: '{x}'"

    elif isinstance(x, (np.ndarray, torch.Tensor)):
        if np.prod(x.shape) <= 20:
            return f"{__repr_name__(x)}: {x.__repr__()}"
        else:
            return f"{__repr_name__(x)}, shape: {x.shape}"

    elif isinstance(x, DeviceView):
        # A device view carries no values of its own: dumping them would require copying the buffer back to the host,
        # so the shape, which is what a caller compares between views, is reported without any of the data
        return f"{__repr_name__(x)}, shape: {x.shape}"

    # elif isinstance(x, (gs.IntEnum, gs.UID)):
    #     return x.__repr__()

    elif x is None:
        return "None"

    else:
        return __repr_name__(x)


def __repr_name__(x):
    """
    Only used for non-genesis object by `brief()`.
    To convert <class 'classname'> into <classname>.
    """
    if isinstance(x, type):
        raw_class_name = str(x)
    else:
        raw_class_name = str(x.__class__)
    full_name = f"<{' '.join(raw_class_name.split(' ')[1:])[1:-2]}>"
    return full_name
