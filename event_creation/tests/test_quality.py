"""Tests for submission/quality/{fr_tests,pal_tests,util}.py.

Each quality check takes an events recarray (wrapped by @as_recarray) and a
``files`` dict; it asserts on the event structure. We synthesize small
structured arrays that both pass and violate each check.
"""
import json

import numpy as np
import pytest

from ..submission.quality import fr_tests, pal_tests, util


EVENT_DTYPE = np.dtype([
    ("type", "U20"), ("list", "i4"), ("intrusion", "i4"),
    ("category", "U20"), ("item_name", "U20"), ("serialpos", "i4"),
    ("probepos", "i4"), ("eegoffset", "i8"), ("mstime", "i8"),
    ("experiment", "U20"),
])


def make_events(rows):
    """rows: list of dicts -> structured ndarray with EVENT_DTYPE defaults."""
    arr = np.zeros(len(rows), dtype=EVENT_DTYPE)
    for i, r in enumerate(rows):
        for k, v in r.items():
            arr[i][k] = v
    return arr


# --------------------------------------------------------------------------- #
# fr_tests
# --------------------------------------------------------------------------- #
def test_catfr_categories_pass():
    ev = make_events([
        {"type": "WORD", "list": 1, "category": "FRUIT", "item_name": "APPLE"},
        {"type": "REC_WORD", "list": 1, "intrusion": 0, "category": "FRUIT"},
    ])
    fr_tests.test_catfr_categories(ev, {})


def test_catfr_categories_fail_missing_word_category():
    ev = make_events([
        {"type": "WORD", "list": 1, "category": "X"},
    ])
    with pytest.raises(AssertionError, match="word presentations missing"):
        fr_tests.test_catfr_categories(ev, {})


def test_session_length_pass_and_fail():
    ev = make_events([{"type": "TRIAL", "list": i} for i in range(5)])
    fr_tests.test_session_length(ev, {})
    too_many = make_events([{"type": "TRIAL", "list": i} for i in range(30)])
    with pytest.raises(AssertionError, match="more than 26 lists"):
        fr_tests.test_session_length(too_many, {})


def test_words_in_wordpool_pass(tmp_path):
    pool = tmp_path / "wordpool.txt"
    pool.write_text("APPLE\nBANANA\n")
    ev = make_events([{"type": "WORD", "list": 1, "item_name": "APPLE"}])
    fr_tests.test_words_in_wordpool(ev, {"wordpool": str(pool)})


def test_words_in_wordpool_fail(tmp_path):
    pool = tmp_path / "wordpool.txt"
    pool.write_text("ORANGE\n")
    ev = make_events([{"type": "WORD", "list": 1, "item_name": "APPLE"}])
    with pytest.raises(AssertionError, match="Wordpool missing"):
        fr_tests.test_words_in_wordpool(ev, {"wordpool": str(pool)})


def test_words_in_wordpool_no_file():
    ev = make_events([{"type": "WORD", "list": 1, "item_name": "APPLE"}])
    # no wordpool key -> function returns without asserting
    fr_tests.test_words_in_wordpool(ev, {})


def _two_lists_words():
    rows = []
    for lst in (1, 2):
        for sp in range(13):
            rows.append({"type": "WORD", "list": lst, "serialpos": sp})
    return make_events(rows)


def test_serialpos_order_pass():
    fr_tests.test_serialpos_order(_two_lists_words(), {})


def test_serialpos_order_fail_nonuniform():
    ev = make_events([
        {"type": "WORD", "list": 1, "serialpos": 0},
        {"type": "WORD", "list": 1, "serialpos": 2},  # jumps by 2
    ])
    with pytest.raises(AssertionError, match="not increasing uniformly"):
        fr_tests.test_serialpos_order(ev, {})


def test_words_per_list_pass():
    fr_tests.test_words_per_list(_two_lists_words(), {})


def test_words_per_list_fail():
    # serialpos 0 appears twice in list 1, once in list 2 -> "repeated"
    ev = make_events([
        {"type": "WORD", "list": 1, "serialpos": 0},
        {"type": "WORD", "list": 1, "serialpos": 0},
        {"type": "WORD", "list": 2, "serialpos": 0},
    ])
    with pytest.raises(AssertionError):
        fr_tests.test_words_per_list(ev, {})


def test_rec_word_position_pass():
    ev = make_events([
        {"type": "REC_START", "list": 1, "eegoffset": 100, "mstime": 100},
        {"type": "REC_WORD", "list": 1, "eegoffset": 150, "mstime": 150},
        {"type": "REC_END", "list": 1, "eegoffset": 200, "mstime": 200},
    ])
    fr_tests.test_rec_word_position(ev, {})


def test_rec_word_position_fail_early():
    ev = make_events([
        {"type": "REC_START", "list": 1, "eegoffset": 100, "mstime": 100},
        {"type": "REC_WORD", "list": 1, "eegoffset": 50, "mstime": 50},
        {"type": "REC_END", "list": 1, "eegoffset": 200, "mstime": 200},
    ])
    with pytest.raises(AssertionError, match="before REC_START"):
        fr_tests.test_rec_word_position(ev, {})


def test_math_position_pass_and_fail():
    ok = make_events([
        {"type": "DISTRACT_START", "list": 1, "eegoffset": 100, "mstime": 100},
        {"type": "PROB", "list": 1, "eegoffset": 150, "mstime": 150},
        {"type": "DISTRACT_END", "list": 1, "eegoffset": 200, "mstime": 200},
    ])
    fr_tests.test_math_position(ok, {})
    bad = make_events([
        {"type": "DISTRACT_START", "list": 1, "eegoffset": 100, "mstime": 100},
        {"type": "PROB", "list": 1, "eegoffset": 50, "mstime": 50},
        {"type": "DISTRACT_END", "list": 1, "eegoffset": 200, "mstime": 200},
    ])
    with pytest.raises(AssertionError, match="before DISTRACT_START"):
        fr_tests.test_math_position(bad, {})


def test_rec_bracket_empty_passes():
    # no lists -> the per-list loop never runs, function returns cleanly
    empty = make_events([{"type": "WORD", "list": 0}])[:0]
    fr_tests.test_rec_bracket(empty, {})


def test_rec_bracket_nonempty_raises_typeerror():
    # NOTE: source calls ``rec_start.any()`` on a *structured* recarray, which
    # numpy cannot cast to bool -> TypeError. This documents that limitation;
    # lines 161-162 are unreachable as a result.
    ev = make_events([{"type": "REC_START", "list": 1}])
    with pytest.raises(TypeError):
        fr_tests.test_rec_bracket(ev, {})


def _stim_config(tmp_path, with_artifact=True):
    cfg = {"experiment": {}}
    if with_artifact:
        cfg["experiment"]["artifact_detection"] = {
            "artifact_detection_number_of_stims_per_channel": 1}
        cfg["experiment"]["experiment_specific_data"] = {"stim_channels": ["A"]}
    p = tmp_path / "experiment_config.json"
    p.write_text(json.dumps(cfg))
    return {"experiment_config": [str(p)]}


def test_stim_on_position_pass(tmp_path):
    files = _stim_config(tmp_path, with_artifact=True)
    ev = make_events([
        {"type": "TRIAL", "list": 1, "mstime": 100, "experiment": "FR3"},
        {"type": "STIM_ON", "list": 1, "mstime": 200, "experiment": "FR3"},
    ])
    fr_tests.test_stim_on_position(ev, files)


def test_stim_on_position_no_stim_events(tmp_path):
    files = _stim_config(tmp_path, with_artifact=True)
    ev = make_events([{"type": "TRIAL", "list": 1, "mstime": 100,
                       "experiment": "FR3"}])
    fr_tests.test_stim_on_position(ev, files)


def test_stim_on_position_keyerror_returns(tmp_path):
    files = _stim_config(tmp_path, with_artifact=False)
    ev = make_events([
        {"type": "TRIAL", "list": 1, "mstime": 100, "experiment": "FR3"},
        {"type": "STIM_ON", "list": 1, "mstime": 200, "experiment": "FR3"},
    ])
    # missing artifact_detection -> KeyError -> early return, no assertion
    fr_tests.test_stim_on_position(ev, files)


def test_stim_on_position_uses_eegoffset_for_fr6(tmp_path):
    files = _stim_config(tmp_path, with_artifact=True)
    ev = make_events([
        {"type": "TRIAL", "list": 1, "eegoffset": 100, "experiment": "FR6"},
        {"type": "STIM_ON", "list": 1, "eegoffset": 200, "experiment": "FR6"},
    ])
    fr_tests.test_stim_on_position(ev, files)


# --------------------------------------------------------------------------- #
# pal_tests
# --------------------------------------------------------------------------- #
def test_pal_session_length_pass_and_fail():
    # pal_tests functions are NOT @as_recarray-wrapped -> pass a recarray
    ev = np.rec.array(make_events([{"type": "TRIAL", "list": i} for i in range(5)]))
    pal_tests.test_session_length(ev, {})
    bad = np.rec.array(make_events([{"type": "TRIAL", "list": i} for i in range(30)]))
    with pytest.raises(AssertionError, match="more than 26 lists"):
        pal_tests.test_session_length(bad, {})


def test_pal_words_per_list_pass():
    rows = []
    for lst in (1, 2):
        for pos in range(1, 4):
            rows.append({"type": "STUDY_PAIR", "list": lst,
                         "serialpos": pos, "probepos": pos})
            rows.append({"type": "REC_EVENT", "list": lst,
                         "serialpos": pos, "probepos": pos})
    ev = np.rec.array(make_events(rows))
    pal_tests.test_words_per_list(ev, {})


# --------------------------------------------------------------------------- #
# util
# --------------------------------------------------------------------------- #
def test_get_time_field_no_event_log():
    assert util.get_time_field({}) == "mstime"


def test_get_time_field_sys4_mstime(tmp_path):
    log = tmp_path / "event.log"
    log.write_text("versions:\n  Ramulator: '4.1'\n")
    assert util.get_time_field({"event_log": [str(log)]}) == "mstime"


def test_get_time_field_sys33_eegoffset(tmp_path):
    log = tmp_path / "event.log"
    log.write_text("versions:\n  Ramulator: '3.3'\n")
    assert util.get_time_field({"event_log": [str(log)]}) == "eegoffset"


def test_get_time_field_old_version_mstime(tmp_path):
    log = tmp_path / "event.log"
    log.write_text("versions:\n  Ramulator: '3.0'\n")
    assert util.get_time_field({"event_log": [str(log)]}) == "mstime"


def test_get_time_field_sys4_no_metadata_first_line(tmp_path):
    # First line is a JSON ELEMEM record (not YAML mapping) -> ParserError ->
    # fallback parses first line, sees type ELEMEM -> version 4.0 -> mstime
    log = tmp_path / "event.log"
    log.write_text('{"type": "ELEMEM", "data": {}}\n{"type": "WORD"}\n')
    assert util.get_time_field({"event_log": [str(log)]}) == "mstime"


def test_get_time_field_parser_error_reraises(tmp_path):
    log = tmp_path / "event.log"
    log.write_text('{"type": "SOMETHING_ELSE"}\nmore: junk\n')
    with pytest.raises(Exception):
        util.get_time_field({"event_log": [str(log)]})


def test_timed_decorator_passes_time_field(tmp_path):
    log = tmp_path / "event.log"
    log.write_text("versions:\n  Ramulator: '4.1'\n")

    @util.timed
    def fn(events, files, time_field=None):
        return time_field

    assert fn(None, {"event_log": [str(log)]}) == "mstime"


def test_as_recarray_wraps_events():
    captured = {}

    @util.as_recarray
    def fn(events, files):
        captured["is_recarray"] = isinstance(events, np.recarray)

    fn(make_events([{"type": "WORD", "list": 1}]), {})
    assert captured["is_recarray"]
