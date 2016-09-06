from __future__ import print_function

# TODO: REPLACE WITH PYTHON LOGGING MODULE

def error_swallower(fn):
    def wrapper(*args, **kwargs):
        if Logger.SWALLOW_ERRORS:
            try:
                return fn(*args, **kwargs)
            except Exception as e:
                try:
                    Logger.last_log._log('SWALLOWING ERROR {}'.format(e.message))
                except Exception:
                    pass
        else:
            return fn(*args, **kwargs)

    return wrapper

class Logger(object):

    SWALLOW_ERRORS = True
    last_log = None

    def __init__(self, do_print=True, swallow_errors=True, label=None, *log_filenames):
        self.do_print = do_print
        self.SWALLOW_ERRORS = swallow_errors
        self.log_files = {log_filename: open(log_filename, 'a') for log_filename in log_filenames}
        self.last_log = self
        self.label = None

    def set_label(self, label):
        self.label = label

    @error_swallower
    def add_log_files(self, *log_filenames):
        for log_filename in log_filenames:
            if log_filename not in self.log_files:
                self.log_files[log_filename] = (open(log_filename, 'a'))
                self.log('Log file {} added'.format(log_filename))
            else:
                self.log('Already logging to {}'.format(log_filename))

    @error_swallower
    def remove_log_files(self, *log_filenames):
        for log_filename in log_filenames:
            if log_filename in self.log_files:
                self.log_files[log_filename].close()
                self.log('Log file {} closed'.format(log_filename))
                del self.log_files[log_filename]
            else:
                self.log('Cannot close log file {}'.format(log_filename), 'ERROR')

    def _log(self, msg, level='INFO', *args, **kwargs):
        log_msg = '{}: {}'.format(level, msg)
        if self.label:
            log_msg = '({}) {}'.format(self.label, log_msg)
        if self.do_print:
            print(log_msg, *args, **kwargs)
        for log_file in self.log_files.values():
            log_file.write(log_msg + '\n')

    @error_swallower
    def log(self, *args, **kwargs):
        self._log(*args, **kwargs)