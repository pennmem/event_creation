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


def describe_recarray(arr):
    names = arr.dtype.names
    lens = [len(name) for name in names]
    padding_max = max(lens) + PPRINT_PADDING
    paddings = [padding_max - this_len for this_len in lens]
    for name, padding in zip(names, paddings):
        shape = arr[name].shape
        print('%s:%s%s' % (name, ' '*padding,shape))


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

def mkdtype(d):
    dtype = {'names' : [], 'formats' : []}
    for k,v in d.items():
        dtype['names'].append(k)
        if isinstance(v, dict):
            dtype['formats'].append(mkdtype(v))
        elif isinstance(v,np.ndarray):
            dtype['formats'].append(dtype)
        elif isinstance(v, int):
            dtype['formats'].append('int64')
        elif isinstance(v, (str, unicode)):
            dtype['formats'].append('S32')
        elif isinstance(v, bool):
            dtype['formats'].append('bool')
        elif isinstance(v, float):
            dtype['formats'].append('float64')
        else:
            raise Exception('Could not convert type %s' % type(v))

    return dtype

def from_json(json_filename):
    d = json.load(open(json_filename))
    if not isinstance(d, list):
        d = [d]
    dt = mkdtype(d[0])
    arr = np.zeros(len(d), dt)
    copy_values(d, arr)
    return arr

def copy_values(dict_list, rec_arr):
    dict_fields = {}
    for k, v, in dict_list[0].items():
        if isinstance(v, dict):
            dict_fields[k] = [inner_dict[k] for inner_dict in dict_list]
    for i, d in enumerate(dict_list):
        for k, v in d.items():
            if k in dict_fields:
                continue

            if isinstance(v, dict):
                copy_values([v], rec_arr[i][k])
            else:
                rec_arr[i][k] = v

    for k, v in dict_fields.items():
        copy_values( v, rec_arr[k])