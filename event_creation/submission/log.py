import os
import logging
from logging.handlers import TimedRotatingFileHandler
import traceback as tb

from . import fileutil
from .configuration import paths


class Logger(object):
    def __init__(self):
        self.label = None  # type: str
        self.subject = None  # type: str
        self.protocol = None  # type: str
        self.subject_handler = None  # type: logging.Handler

        self._logger = logging.getLogger('submission')
        self._logger.setLevel(logging.DEBUG)

        self.formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

        self.stdout_handler = logging.StreamHandler()
        self.stdout_handler.setLevel(logging.INFO)
        self.stdout_handler.setFormatter(self.formatter)
        self._logger.addHandler(self.stdout_handler)

        # Creates a new log file every 30 days
        if not os.path.exists(os.path.join(paths.db_root, 'protocols')):
            fileutil.makedirs(os.path.join(paths.db_root, 'protocols'))

        # self.master_file_handler = logging.handlers.TimedRotatingFileHandler(
            # os.path.join(paths.db_root, 'protocols', 'log.txt'),
            # 'D', 30, backupCount=1)
        self.master_file_handler = logging.FileHandler(os.path.join(paths.db_root, 'protocols', 'log.txt'))
        self.master_file_handler.setLevel(logging.INFO)
        self.master_file_handler.setFormatter(self.formatter)
        self._logger.addHandler(self.master_file_handler)

    def set_stdout_level(self, level):
        """Set the log level for the stream handler."""
        self.stdout_handler.setLevel(level)

    def _set_subject_handler(self):
        """Add a file handler to log to the subject directory."""
        if self.subject_handler:
            self._logger.removeHandler(self.subject_handler)
        filename = os.path.join(paths.db_root,
                                'protocols', self.protocol,
                                'subjects', self.subject,
                                'log.txt')
        if not os.path.exists(os.path.dirname(filename)):
            fileutil.makedirs(os.path.dirname(filename))
        self.subject_handler = logging.FileHandler(filename)
        self.subject_handler.setLevel(logging.DEBUG)
        self.subject_handler.setFormatter(self.formatter)
        self._logger.addHandler(self.subject_handler)
        self.debug("Log file {} opened".format(filename))

    def set_label(self, label):
        """Set the label to be attached to log messages."""
        self.label = label

    def set_subject(self, subject, protocol):
        """Set the subject to be attached to log messages."""
        self.subject = subject
        self.protocol = protocol
        self._set_subject_handler()

    def unset_subject(self):
        """Unset the current subject."""
        self.subject = None
        self.protocol = None
        if self.subject_handler:
            self._logger.removeHandler(self.subject_handler)
            self.subject_handler = None

    @staticmethod
    def _format_msg(msg, **kwargs):
        return '{subject} - {label} - {msg}'.format(msg=msg, **kwargs)

    def debug(self, msg):
        self._logger.debug(self._format_msg(msg, subject=self.subject, label=self.label))

    def info(self, msg):
        self._logger.info(self._format_msg(msg, subject=self.subject, label=self.label))

    def warn(self, msg):
        self._logger.warn(self._format_msg(" ****** " + msg, subject=self.subject, label=self.label))

    def error(self, msg):
        self._logger.error(self._format_msg(" ************ " + msg, subject=self.subject, label=self.label))

    def critical(self, msg):
        self._logger.critical(self._format_msg(" ********************* " + msg, subject=self.subject, label=self.label))

try:
    logger = Logger()
except Exception as e:
    tb.print_exc(e)
    logger = None
