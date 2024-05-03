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
        try:
            with open(files['event_log'][0]) as event_log_file:
                event_log = yaml.load(event_log_file, Loader=yaml.FullLoader)       # specify Loader
                version_no = event_log['versions']['Ramulator']
        except yaml.parser.ParserError as e:
            with open(files['event_log'][0]) as event_log_file:
                event_log = yaml.load(event_log_file.readline(), Loader=yaml.FullLoader)
                if event_log["type"] == 'ELEMEM' or event_log["type"] == 'EEGSTART':  # some sys4 no metadata 1st line
                    version_no = '4.0'
                else:
                    raise e
        if version_no >= '4.0':
            time_field = 'mstime'     # use mstime field for system 4 (helps with eeg replacement checks)
        elif version_no >= '3.3' and version_no < '4.0':
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
