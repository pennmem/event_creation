import pprint
import numpy as np

PPRINT_PADDING = 2

def pprint_rec(arr):
    print(pformat_rec(arr))

def pformat_rec(arr, init_indent = 0):
    names = arr.dtype.names
    lens = [len(name) for name in names]
    padding_max = max(lens) + PPRINT_PADDING
    paddings = [padding_max - this_len for this_len in lens]
    formatted = []
    init_padding = ' '*init_indent
    for i, name in enumerate(names):
        value = arr[name]
        if isinstance(value, np.ndarray) and value.dtype.names:
            formatted_value = '\n' + pformat_rec(value, init_indent = padding_max+1)
        else:
            formatted_value = _format_and_indent(value, padding_max+PPRINT_PADDING)
        formatted.append('%s%s:%s%s' % (init_padding, name, ' '*paddings[i], formatted_value))
    return '\n'.join(formatted)

def _format_and_indent(input, indent):
    formatted = pprint.pformat(input)
    stripped = [x.strip() for x in formatted.split('\n')]
    return ('\n'+' '*indent).join(stripped)
