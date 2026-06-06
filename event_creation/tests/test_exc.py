"""Tests for submission/exc.py — every custom exception is an Exception
subclass and can be raised/caught with a message."""
import pytest

from ..submission import exc


EXCEPTIONS = [
    exc.TransferError, exc.AlignmentError, exc.LogParseError,
    exc.UnknownExperimentError, exc.EventFieldError, exc.ConfigurationError,
    exc.WebAPIError, exc.MontageError, exc.NoEventsError,
    exc.NoAnnotationError, exc.ProcessingError, exc.EEGError,
    exc.PeakFindingError,
]


@pytest.mark.parametrize("exc_cls", EXCEPTIONS)
def test_exception_subclass_and_raisable(exc_cls):
    assert issubclass(exc_cls, Exception)
    with pytest.raises(exc_cls) as info:
        raise exc_cls("boom")
    assert "boom" in str(info.value)
