"""Toggleable function-call tracer for event_creation.

A lightweight, source-free alternative to hand-inserted ``print()`` lines: it
uses :func:`sys.settrace` to print every function call that happens *inside the
event_creation package* (stdlib / numpy / pandas / mne frames are filtered out),
so you can see exactly which functions — e.g. which System-4 aligner — actually
run during a session import.

Usage
-----
Programmatic on/off::

    from event_creation import _tracer
    _tracer.on()
    run_session_import(...)
    _tracer.off()

Env-var auto-enable (no code edits) — the runner scripts (submit.py,
automatic_run.py, ...) check ``EC_TRACE`` and call :func:`on` for you::

    EC_TRACE=1 python submit.py ...

Each call prints::

    >>> TRACE submission/alignment/system4.py:37 align

Limitations
-----------
``sys.settrace`` only traces frames created *after* it is installed, and only on
the thread that called it. That covers a normal synchronous run started right
after :func:`on`. If event_creation is ever made to spawn worker threads, also
register :func:`threading.settrace` with the same hook.

Note: when run under ``automatic_run.py``, stdout is redirected to the per-session
logfile (``~/logs/automatic_run_<exp>_<sub>_<sess>.log``), so the ``>>> TRACE``
lines land there, not on the console.
"""
import os
import sys

# Package root = directory containing this file. Only frames whose filename
# lives under here get traced.
_PKG = os.path.dirname(os.path.abspath(__file__))

_active = False


def _trace(frame, event, arg):
    # Only care about function entry. Returning the hook keeps it active for the
    # frame; returning None would stop local tracing of that frame (fine for us,
    # but returning the hook lets nested calls report too via the global hook).
    if event != "call":
        return _trace
    code = frame.f_code
    filename = code.co_filename
    if not filename.startswith(_PKG):
        return  # not our code — don't descend into stdlib/3rd-party frames
    rel = os.path.relpath(filename, _PKG)
    print(">>> TRACE %s:%d %s" % (rel, code.co_firstlineno, code.co_name))
    return _trace


def on():
    """Start printing the event_creation call trace to stdout."""
    global _active
    sys.settrace(_trace)
    _active = True


def off():
    """Stop tracing."""
    global _active
    sys.settrace(None)
    _active = False


def is_active():
    return _active
