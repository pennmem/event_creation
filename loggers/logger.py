from __future__ import print_function
import logging
import os
import logging.handlers
from submission.config import DB_ROOT

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
    MSG_FORMAT = '{subject} - {label} - {msg}'

    def __init__(self, do_print=True, swallow_errors=True, label=None):
        self.do_print = do_print
        self.SWALLOW_ERRORS = swallow_errors
        self.last_log = self
        self.label = None
        self.subject = None
        self.protocol = None

        self._logger = logging.getLogger('submission')
        self._logger.setLevel(logging.DEBUG)

        self.formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

        self.stdout_handler = logging.StreamHandler()
        self.stdout_handler.setLevel(logging.INFO)
        self.stdout_handler.setFormatter(self.formatter)
        self._logger.addHandler(self.stdout_handler)

        # Creates a new log file every 30 days
        if not os.path.exists(os.path.join(DB_ROOT, 'protocols')):
            os.makedirs(os.path.join(DB_ROOT, 'protocols'))

        self.master_file_handler = logging.handlers.TimedRotatingFileHandler(
            os.path.join(DB_ROOT, 'protocols', 'log.txt'), 'D', 30)
        self.master_file_handler.setLevel(logging.INFO)
        self.master_file_handler.setFormatter(self.formatter)
        self._logger.addHandler(self.master_file_handler)

        self.subject_handler = None

    def set_stdout_level(self, level):
        self.stdout_handler.setLevel(level)

    def set_subject_handler(self):
        if self.subject_handler:
            self._logger.removeHandler(self.subject_handler)
        filename = os.path.join(DB_ROOT,
                                'protocols', self.protocol,
                                'subjects', self.subject,
                                'log.txt')
        if not os.path.exists(os.path.dirname(filename)):
            os.makedirs(os.path.dirname(filename))
        self.subject_handler = logging.FileHandler(filename)
        self.subject_handler.setLevel(logging.DEBUG)
        self.subject_handler.setFormatter(self.formatter)
        self._logger.addHandler(self.subject_handler)
        self.debug("Log file {} opened".format(filename))

    def set_label(self, label):
        self.label = label

    def set_subject(self, subject, protocol):
        self.subject = subject
        self.protocol = protocol
        self.set_subject_handler()

    def unset_subject(self):
        self.subject = None
        self.protocol = None
        if self.subject_handler:
            self._logger.removeHandler(self.subject_handler)
            self.subject_handler = None

    def format_msg(self, msg, **kwargs):
        return self.MSG_FORMAT.format(msg=msg, **kwargs)

    def debug(self, msg):
        self._logger.debug(self.format_msg(msg, subject=self.subject, label=self.label))

    def info(self, msg):
        self._logger.info(self.format_msg(msg, subject=self.subject, label=self.label))

    def warn(self, msg):
        self._logger.warn(self.format_msg(msg, subject=self.subject, label=self.label))

    def error(self, msg):
        self._logger.error(self.format_msg(msg, subject=self.subject, label=self.label))

    def critical(self, msg):
        self._logger.critical(self.format_msg(msg, subject=self.subject, label=self.label))