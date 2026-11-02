"""
System 1 alignment for the PAL1/IPAL1 task: the lab's System1Aligner, given the task's
pulse times on the clock its events are on.

System1Aligner reads a Unity task's pulse times as the ``time`` of its syncPulse lines,
which UnityEPL tasks write as Unix ms. The PAL1/IPAL1 task's pal_events.jsonl writes
``time`` as a stopwatch that restarts at each launch, and the parser's ``mstime`` is the
wall clock (LAUNCH.epochUnixMs + time; see pal_task_log.py). This class only reads the
pulses on that same wall clock; the rest is System1Aligner's: one EEG file
(sources.json with one entry), one ``.sync.txt``, one line through the pulse windows
matched at the start and the end of the session.
"""

from .system1 import System1Aligner
from ..parsers.pal_task_log import read_task_log, SYNC_PULSE


class TaskLogSystem1Aligner(System1Aligner):
    """System1Aligner with the pulse times read from pal_events.jsonl. See the module docstring."""

    def get_task_pulses(self):
        return [m['_abs'] for launch in read_task_log(self.unity_log_file)
                for m in launch['messages'] if m['type'] == SYNC_PULSE]
