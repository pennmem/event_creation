
def timed(timed_function):
    import yaml
    from functools import wraps

    @wraps(timed_function)
    def wrapped(events,files):
        if 'event_log' in files:
            with open(files['event_log'][0]) as event_log_file:
                event_log = yaml.load(event_log_file)
            version_no = event_log['versions']['Ramulator']
            if version_no>='3.3':
                time_field='eegoffset'
            else:
                time_field = 'mstime'
        else:
            time_field = 'mstime'
        return timed_function(events,files,time_field=time_field)

    return wrapped
