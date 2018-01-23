import pprint
import numpy as np
import json
import numpy
import unicodedata
from collections import defaultdict
import re
from collections import OrderedDict


PPRINT_PADDING = 2


def pprint_rec(arr, recurse=True):
    print(pformat_rec(arr, recurse))


def pformat_rec(arr, recurse=True, init_indent=0):
    names = sorted(arr.dtype.names)
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
    if arr.ndim == 0:
        return {}

    arr_as_dict = []
    names_without_remove = [name for name in arr.dtype.names if name != '_remove']
    for x in arr:
        if (not '_remove' in x.dtype.names) or (not x['_remove']):
            entry = {}
            for name in names_without_remove:
                entry[name] = x[name]
            arr_as_dict.append(entry)

    if len(arr_as_dict) == 0:
        return arr_as_dict

    recarray_keys = []
    array_keys = []
    for key, value in arr_as_dict[0].items():
        if isinstance(value, (np.ndarray, np.record)) and value.dtype.names:
            recarray_keys.append(key)
        elif isinstance(value, (np.ndarray)):
            array_keys.append(key)

    for key in recarray_keys:
        for entry in arr_as_dict:
            if entry[key].size > 1:
                entry[key] = to_dict(entry[key])
            else:
                entry[key] = dict(zip(entry[key].dtype.names, entry[key]))

    for key in array_keys:
        for entry in arr_as_dict:
            entry[key] = list(entry[key])


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
        json.dump(to_dict(arr), fp, cls=MyEncoder, indent=2, sort_keys=True)
    else:
        return json.dumps(to_dict(arr), cls=MyEncoder, indent=2, sort_keys=True)

def get_element_dtype(element):
    if isinstance(element, dict):
        return mkdtype(element)
    elif isinstance(element, int):
        return 'int64'
    elif isinstance(element, (str, unicode)):
        return 'S64'
    elif isinstance(element, bool):
        return 'b'
    elif isinstance(element, float):
        return 'float64'
    elif isinstance(element, list):
        return get_element_dtype(element[0])
    else:
        raise Exception('Could not convert type %s' % type(element))

def mkdtype(d):
    if isinstance(d, list):
        dtype = mkdtype(d[0])
        return dtype
    dtype = []

    for k,v in d.items():
        dtype.append((str(k), get_element_dtype(v)))

    return np.dtype(dtype)

def from_json_old(json_filename):
    d = json.load(open(json_filename))
    if not isinstance(d, list):
        d = [d]
    dt = mkdtype(d[0])
    arr = np.zeros(len(d), dt)
    copy_values(d, arr)
    return arr.view(np.recarray)


def from_jsons(jsons):
    d = json.loads(jsons)
    return from_dict(d)

def from_json(json_filename):
    d = json.load(open(json_filename))
    return from_dict(d)

def from_dict(d):
    if not isinstance(d, list):
        d = [d]

    list_names = []

    for k, v in d[0].items():
        if isinstance(v, list):
            list_names.append(k)

    list_info = defaultdict(lambda *_: {'len': 0, 'dtype': None})

    for entry in d:
        for k in list_names:
            list_info[k]['len'] = max(list_info[k]['len'], len(entry[k]))
            if not list_info[k]['dtype'] and len(entry[k]) > 0:
                if isinstance(entry[k][0], dict):
                    list_info[k]['dtype'] = mkdtype(entry[k][0])
                else:
                    list_info[k]['dtype'] = get_element_dtype(entry[k])

    dtypes_dict = OrderedDict()
    for entry in d:
        for k, v in entry.items():
            if v or (k not in dtypes_dict):
                if k not in list_info:
                    dtypes_dict[str(k)] = get_element_dtype(v)
                else:
                    dtypes_dict[str(k)] =  (list_info[k]['dtype'], list_info[k]['len'])

    dtypes = [(k,) + v if isinstance(v,tuple) else (k,v) for (k,v) in dtypes_dict.iteritems()]

    if dtypes:
        arr = np.zeros(len(d), dtypes).view(np.recarray)
        copy_values(d, arr, list_info)
    else:
        arr = np.array([])
    return arr.view(np.recarray)

def copy_values(dict_list, rec_arr, list_info=None):
    if len(dict_list) == 0:
        return

    dict_fields = {}
    for k, v, in dict_list[0].items():
        if isinstance(v, dict):
            dict_fields[k] = [inner_dict[k] for inner_dict in dict_list]

    for i, sub_dict in enumerate(dict_list):
        for k, v in sub_dict.items():
            if k in dict_fields or  list_info and k in list_info:
                continue

            if isinstance(v, dict):
                copy_values([v], rec_arr[i][k])
            elif isinstance(v, basestring):
                rec_arr[i][k] = strip_accents(v)
            else:
                rec_arr[i][k] = v

    for i, sub_dict in enumerate(dict_list):
        for k,v in sub_dict.items():
            if list_info and k in list_info:
                arr = np.zeros(list_info[k]['len'], list_info[k]['dtype'])
                if len(v) > 0:
                    if isinstance(v[0], dict):
                        copy_values(v, arr)
                    else:
                        for j, element in enumerate(v):
                            arr[j] = element

                rec_arr[i][k] = arr.view(np.recarray)

    for k, v in dict_fields.items():
        copy_values( v, rec_arr[k])

def strip_accents(s):
    try:
        return str(''.join(c for c in unicodedata.normalize('NFD', unicode(s))
                      if unicodedata.category(c) != 'Mn'))
    except UnicodeError: # If accents can't be converted, just remove them
        return str(re.sub(r'[^A-Za-z0-9 -_.]', '', s))

