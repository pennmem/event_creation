"""Tests for parsers/math_parser.py.

Drives MathSessionLogParser end-to-end on a synthetic math.log (the legacy
tab-delimited pyepl format), which also exercises a large part of the
BaseSessionLogParser / BaseLogParser machinery (primary-log reading, the
type->handler dispatch in parse(), event_default, _add_fields, clean_events).
"""
import numpy as np
import pytest

from ..submission.parsers.math_parser import MathLogParser, MathSessionLogParser
from ..submission.exc import UnknownExperimentError


MATH_LOG = "\n".join([
    "1436709716398\t0\tB\tLogging Begins",
    "1436709791369\t0\tSTART",
    "1436709791385\t1\tPROB\t'3 + 2 + 5 = '\t'10'\t1\t5776\t1",
    "1436709797186\t1\tPROB\t'4 + 3 + 6 = '\t'13'\t1\t2880\t1",
    "1436709816002\t1\tSTOP",
    "1436709899306\t0\tSTART",
    "1436709899322\t1\tPROB\t'7 + 4 + 2 = '\t'13'\t1\t5327\t1",
    "1436709919921\t1\tSTOP",
]) + "\n"


@pytest.fixture
def math_log_file(tmp_path):
    p = tmp_path / "math.log"
    p.write_text(MATH_LOG)
    return str(p)


def _parser(math_log_file, protocol="r1"):
    files = {"math_log": math_log_file}
    return MathSessionLogParser(protocol, "R1999X", 0, "FR1", 0, files)


def test_math_session_parse_basic(math_log_file):
    events = _parser(math_log_file).parse()
    # 3 PROB + 2 START + 2 STOP = 7 events (B is skipped, first empty dropped)
    assert len(events) == 7
    probs = events[events["type"] == "PROB"]
    assert len(probs) == 3
    # first PROB: 3 + 2 + 5, answer 10, correct, rectime 5776
    first = probs[0]
    assert list(first["test"]) == [3, 2, 5]
    assert first["answer"] == 10
    assert first["iscorrect"] == 1
    assert first["rectime"] == 5776


def test_math_session_list_numbering(math_log_file):
    events = _parser(math_log_file).parse()
    # START list logic: -999 -> -1 (first), -1 -> 1 (second)
    starts = events[events["type"] == "START"]
    assert list(starts["list"]) == [-1, 1]
    # PROBs inherit the current list value
    probs = events[events["type"] == "PROB"]
    assert list(probs["list"]) == [-1, -1, 1]


def test_math_session_clean_events_sorts_and_recarrays(math_log_file):
    parser = _parser(math_log_file)
    events = parser.clean_events(parser.parse())
    # clean_events returns a recarray sorted by mstime
    assert isinstance(events, np.recarray)
    assert (np.diff(events.mstime) >= 0).all()
    assert (events.experiment == "FR1").all()


def test_math_session_ltp_adds_eog_field(tmp_path):
    p = tmp_path / "math.log"
    p.write_text(MATH_LOG)
    files = {"math_log": str(p)}
    parser = MathSessionLogParser("ltp", "LTP999", 0, "ltpFR", 0, files)
    events = parser.parse()
    assert "eogArtifact" in events.dtype.names


def test_math_fields_classmethods():
    f = MathSessionLogParser._math_fields()
    assert any(name == "test" for name, *_ in f)
    f_ltp = MathSessionLogParser._math_fields_ltp()
    assert any(name == "eogArtifact" for name, *_ in f_ltp)


# --------------------------------------------------------------------------- #
# MathLogParser factory dispatch
# --------------------------------------------------------------------------- #
def test_factory_dispatches_math_log(math_log_file):
    parser = MathLogParser("r1", "R1999X", 0, "FR1", 0,
                           {"math_log": math_log_file})
    assert isinstance(parser, MathSessionLogParser)


def test_factory_unknown_raises():
    with pytest.raises(UnknownExperimentError):
        MathLogParser("r1", "R1999X", 0, "FR1", 0, {"something_else": "x"})
