"""Rhino-backed smoke tests against real session logs on /data10 and /data/eeg.

These exercise the real file-discovery + parsing paths (the hard-coded
/data10 globs in get_heart, real pyepl math.log parsing) that the synthetic
unit tests stub out. They are skipped automatically off-rhino so the rest of
the suite still runs anywhere.
"""
import os

import numpy as np
import pytest

needs_data10 = pytest.mark.skipif(
    not os.path.exists("/data10/RAM/subjects"),
    reason="requires rhino /data10 data")
needs_dataeeg = pytest.mark.skipif(
    not os.path.exists("/data/eeg"),
    reason="requires rhino /data/eeg data")


@needs_data10
def test_get_heart_real_nonbroken_session():
    from heartbeat_correction import fix_heartbeats_sys4 as hb
    # R1700E / RepFR2 / session_0 is the documented non-broken example
    task = hb.get_heart("R1700E", "RepFR2", 0, load_host_pc=False)
    host = hb.get_heart("R1700E", "RepFR2", 0, load_host_pc=True)
    assert len(task) > 100 and len(host) > 100
    assert (task["hardware_system"] == "task_laptop").all()
    assert (host["hardware_system"] == "host_pc").all()


@needs_data10
def test_get_heart_real_correction_runs():
    from heartbeat_correction import fix_heartbeats_sys4 as hb
    task = hb.get_heart("R1700E", "RepFR2", 0, drop_network_test=True,
                        load_host_pc=False)
    host = hb.get_heart("R1700E", "RepFR2", 0, drop_network_test=True,
                        load_host_pc=True)
    task["session"] = 0
    host["session"] = 0
    import pandas as pd
    hb_all = pd.concat([task, host], ignore_index=True)
    res = hb.get_heartbeat_correction(hb_all, ignore_errors=True)
    # slope should be very close to 1 for a real, well-aligned session
    assert abs(res["slope"] - 1.0) < 1e-3


@needs_data10
def test_get_heart_real_broken_session_raises():
    from heartbeat_correction import fix_heartbeats_sys4 as hb
    # FBG490 logged only FNSB syncbox heartbeats on the task side -> no
    # task<->elemem HEARTBEATs -> ValueError
    with pytest.raises(ValueError, match="No HEARTBEAT"):
        hb.get_heart("FBG490", "EFRCourierOpenLoop", 0, load_host_pc=False)


@needs_dataeeg
def test_math_parser_real_log():
    from ..submission.parsers.math_parser import MathSessionLogParser
    log = "/data/eeg/R1111M/behavioral/FR1/session_0/math.log"
    if not os.path.exists(log):
        pytest.skip("R1111M math.log not present")
    parser = MathSessionLogParser("r1", "R1111M", 0, "FR1", 0,
                                  {"math_log": log})
    events = parser.clean_events(parser.parse())
    assert len(events) > 0
    probs = events[events.type == "PROB"]
    assert len(probs) > 0
    # parsed test problems are 3-element integer arrays
    assert probs[0].test.shape == (3,)
    assert (np.diff(events.mstime) >= 0).all()
