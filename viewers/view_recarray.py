import pprint
import numpy as np
import json
import numpy

PPRINT_PADDING = 2


def pprint_rec(arr, recurse=True):
    print(pformat_rec(arr, recurse))


def pformat_rec(arr, recurse=True, init_indent=0):
    names = arr.dtype.names
    lens = [len(name) for name in names]
    padding_max = max(lens) + PPRINT_PADDING
    paddings = [padding_max - this_len for this_len in lens]
    formatted = []
    init_padding = ' '*init_indent
    for i, name in enumerate(names):
        value = arr[name]
        if recurse and isinstance(value, (np.ndarray, np.record)) and value.dtype.names:
            formatted_value = '\n' + pformat_rec(value, recurse, init_indent=padding_max+1)
        else:
            formatted_value = _format_and_indent(value, init_indent + padding_max + PPRINT_PADDING)
        formatted.append('%s%s:%s%s' % (init_padding, name, ' '*paddings[i], formatted_value))
    return '\n'.join(formatted)


def _format_and_indent(this_input, indent):
    formatted = pprint.pformat(this_input)
    stripped = [x.strip() for x in formatted.split('\n')]
    return ('\n'+' '*indent).join(stripped)


def to_dict(arr):
    arr_as_dict = [dict(zip(arr.dtype.names, x)) for x in arr]
    if len(arr_as_dict) == 0:
        return arr_as_dict

    keys_to_convert = []
    for key, value in arr_as_dict[0].items():
        if isinstance(value, (np.ndarray, np.record)):
            keys_to_convert.append(key)

    for key in keys_to_convert:
        for entry in arr_as_dict:
            if entry[key].size > 1:
                entry[key] = to_dict(entry[key])
            else:
                entry[key] = dict(zip(entry[key].dtype.names, entry[key]))

    return arr_as_dict

class MyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, numpy.integer):
            return int(obj)
        elif isinstance(obj, numpy.floating):
            return float(obj)
        elif isinstance(obj, numpy.ndarray):
            return obj.tolist()
        elif isinstance(obj, numpy.bool_):
            return bool(obj)
        else:
            return super(MyEncoder, self).default(obj)

def to_json(arr, fp=None):
    if fp:
        json.dump(to_dict(arr), fp, cls=MyEncoder, indent=4)
    else:
        return json.dumps(to_dict(arr), cls=MyEncoder, indent=4)
