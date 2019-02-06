import numpy as np
from functools import wraps


def timed(timed_function):

    @wraps(timed_function)
    def wrapped(events,files):
        time_field = get_time_field(files)
        return timed_function(events,files,time_field=time_field)

    return wrapped


def get_time_field(files):
    import yaml
    if 'event_log' in files:
        with open(files['event_log'][0]) as event_log_file:
            event_log = yaml.load(event_log_file)
        version_no = event_log['versions']['Ramulator']
        if version_no >= '3.3':
            time_field = 'eegoffset'
        else:
            time_field = 'mstime'
    else:
        time_field = 'mstime'
    return time_field


def as_recarray(function):

    @wraps(function)
    def wrapped(events,files):
        return function(np.rec.array(events),files)
    return wrapped