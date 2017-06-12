class TransferError(Exception):
    """Raised when files can't be transfered."""


class AlignmentError(Exception):
    """Raised when there is an error aligning EEG data."""


class LogParseError(Exception):
    """Raised when a line in a log file can't be parsed."""


class UnknownExperimentError(Exception):
    """Raised when an unknown experiment is encountered."""


class EventFieldError(Exception):
    """Raised when event fields have errors."""


class ConfigurationError(Exception):
    """Raised when a config file is malformed."""


class MontageError(Exception):
    """Raised when there are montage errors."""


class NoEventsError(Exception):
    """Raised when no events are found."""


class ProcessingError(Exception):
    """Raised when something cannot be processed."""


class EEGError(Exception):
    """Raised when there are errors in EEG processing."""


class PeakFindingError(Exception):
    """Raised when there are errors peak finding."""
