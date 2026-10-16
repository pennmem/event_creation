"""
PAL-U: word-number paired associates with typed answers.

The task (pennmem IPAL1_UnityEPL, experiment PAL-U) shows a word with a
two-digit number and later the word alone; the participant types the number into two
boxes and confirms with Enter. Nothing is recorded or annotated: the answers are in the
log. Beside the PAL1/IPAL1 messages the task sends

* ``RESPONSE_KEY {slot, key, typed, accepted}`` for every key at a probe,
* ``RESPONSE {slot, serialpos, probe, expected, answer, error, points, correct,
  rt_first_key_ms, answer_complete_ms, rt_answer_ms, rt_ms}`` when Enter confirms,
* ``FEEDBACK {points, max_points, session_points, practice}`` when the end-of-list screen
  appears, and ``FEEDBACK_END`` when Enter closes it.

The PAL1/IPAL1 parsers' message handling, restart rule and clocks are reused unchanged
(``ElememPALLogParser`` for System 4, ``PALTaskLogParser`` for System 1); this module
replaces only the scoring, which reads RESPONSE instead of ``.ann`` files.

The moment of recall
--------------------
A probe's REC_EVENT is placed when the second digit of the confirmed answer was typed:
the last RESPONSE_KEY of that probe, before its RESPONSE, that filled the second box. A
Backspace from two digits and a new second digit moves it; Enter does not. Its ``mstime``
is that key message's own time on the EEG clock, mapped as every other message is.

Fields added to the PAL schema
------------------------------
``number`` (the pair's number), ``resp_number`` (the answer), ``num_error``
(answer - number), ``points`` (100 x e^(-|error|/6), as the task scored it),
``rt_first_key`` and ``rt_enter`` (ms from the probe's onset to the first digit and to
Enter). ``RT`` is ms from the probe's onset to the moment of recall. ``correct`` is an
exact answer. ``resp_word`` holds the answer as digits, and ``intrusion`` treats the
answer like a word: 0 if it is another pair's number in the same list, k if it was last
shown k lists back, -1 otherwise.
"""

import warnings

import numpy as np

from .elemem_pal_parser import ElememPALLogParser
from .pal_base import test_session_length, test_one_pair_per_position, test_one_probe_per_slot
from .pal_tasklog_parser import PALTaskLogParser


def test_every_probe_answered(events, files):
    probes = events[(events.type == 'TEST_PROBE') & (events.list > 0)]
    assert (probes.correct != -999).all(), '%d probes have no answer' % (
        (probes.correct == -999).sum())


def _int(v, default=-999):
    try:
        return int(round(float(v)))
    except (TypeError, ValueError):
        return default


class PALUMixin(object):
    """PAL-U scoring on top of a PAL parser. Must come before it in the bases."""

    _TESTS = [test_session_length, test_one_pair_per_position, test_one_probe_per_slot,
              test_every_probe_answered]

    @classmethod
    def _pal_fields(cls):
        return super(PALUMixin, cls)._pal_fields() + (
            ('number', -999, 'int16'),
            ('resp_number', -999, 'int16'),
            ('num_error', -999, 'int16'),
            ('points', -999, 'int16'),
            ('rt_first_key', -999, 'int32'),
            ('rt_enter', -999, 'int32'),
        )

    def _init_pal(self):
        super(PALUMixin, self)._init_pal()
        self._keys = {}          # (list, probepos) -> [RESPONSE_KEY messages]
        self._responses = {}     # (list, probepos) -> RESPONSE message
        self._add_type_to_new_event(
            response_key=self.event_response_key,
            response=self.event_response,
            feedback=self.event_feedback,
            feedback_end=self.event_feedback_end,
        )

    def event_study_pair(self, msg):
        event = super(PALUMixin, self).event_study_pair(msg)
        event.number = _int(msg['data'].get('number', msg['data'].get('word2')))
        return event

    def event_test_probe(self, msg):
        event = super(PALUMixin, self).event_test_probe(msg)
        event.number = _int(msg['data'].get('number', msg['data'].get('expecting')))
        return event

    def event_response_key(self, msg):
        pp = _int(msg['data'].get('slot')) + 1
        self._keys.setdefault((self._list, pp), []).append(msg)
        return False

    def event_response(self, msg):
        pp = _int(msg['data'].get('slot')) + 1
        self._responses[(self._list, pp)] = msg
        event = self.event_default(msg)
        event.type = 'RESPONSE_ENTER'
        event.probepos = pp
        return event

    def event_feedback(self, msg):
        event = self.event_default(msg)
        event.type = 'FEEDBACK_START'
        return event

    def event_feedback_end(self, msg):
        event = self.event_default(msg)
        event.type = 'FEEDBACK_END'
        return event

    def _completion(self, lst, pp, response):
        """The key message that typed the second digit of the confirmed answer."""
        done = None
        for k in self._keys.get((lst, pp), []):
            if float(k.get('_mstime', k['time'])) > float(response.get('_mstime', response['time'])):
                break
            d = k['data']
            key, typed = str(d.get('key', '')), str(d.get('typed', ''))
            if key.isdigit() and len(typed) == 2 and d.get('accepted', True):
                done = k
            elif key == 'backspace' and len(typed) < 2:
                done = None
        return done

    def _score_all(self, events):
        new_events = []
        for (lst, pp), p in self._probes.items():
            if p['serialpos'] == -999:
                warnings.warn('list %d probe %d has no TEST_PROBE; not scored' % (lst, pp))
                continue
            r = self._responses.get((lst, pp))
            if r is None:
                warnings.warn('list %d probe %d has no RESPONSE; left unscored' % (lst, pp))
                continue
            d = r['data']
            answer = _int(d.get('answer'))
            number = _int(p['expecting'])
            error = answer - number
            done = self._completion(lst, pp, r)
            if done is None:
                warnings.warn('list %d probe %d: no key completed the answer; the moment '
                              'of recall is taken from Enter' % (lst, pp))
                done = r
            recall_ms = float(done.get('_mstime', done['time']))

            # RT: probe onset to the moment of recall, on the task's own clock where
            # both times are known (as PAL's RT), else on the EEG clock.
            done_task = done.get('data', {}).get('task_time_ms')
            if p.get('probe_task_ms') is not None and done_task is not None:
                rt = _int(float(done_task) - float(p['probe_task_ms']))
            elif p.get('probe_mstime') is not None:
                rt = _int(recall_ms - float(p['probe_mstime']))
            else:
                rt = _int(d.get('rt_answer_ms'))
            rt_first, rt_enter = _int(d.get('rt_first_key_ms')), _int(d.get('rt_ms'))

            points = _int(d.get('points'))
            correct = int(error == 0)
            word = str(answer)
            intrusion = 0 if correct else self._intrusion(word, lst)

            pair = (events.list == lst) & (events.serialpos == p['serialpos'])
            events.number[pair] = number
            events.resp_word[pair] = word
            events.resp_number[pair] = answer
            events.num_error[pair] = error
            events.points[pair] = points
            events.correct[pair] = correct
            events.intrusion[pair] = intrusion
            events.vocalization[pair] = 0
            events.resp_pass[pair] = 0
            events.RT[pair] = rt
            events.rt_first_key[pair] = rt_first
            events.rt_enter[pair] = rt_enter

            ev = self._empty_event
            ev.type = 'REC_EVENT'
            ev.mstime = int(round(recall_ms))
            ev.list = lst
            ev.session = self._session
            ev.phase = 'PRACTICE' if lst == -1 else 'RETRIEVAL'
            ev.serialpos, ev.probepos = p['serialpos'], pp
            ev.probe_word, ev.expecting_word = p['probe'], p['expecting']
            ev.cue_direction = p['direction']
            ev.study_1, ev.study_2 = p['study_1'], p['study_2']
            ev.number, ev.resp_number, ev.num_error = number, answer, error
            ev.resp_word, ev.points, ev.correct = word, points, correct
            ev.intrusion, ev.vocalization, ev.resp_pass = intrusion, 0, 0
            ev.RT, ev.rt_first_key, ev.rt_enter = rt, rt_first, rt_enter
            ev.exp_version = self._version
            new_events.append(ev)

        if new_events:
            events = np.concatenate([events] + [np.atleast_1d(e) for e in new_events]
                                    ).view(np.recarray)
        return events


class ElememPALULogParser(PALUMixin, ElememPALLogParser):
    """PAL-U on System 4: Elemem's event.log. See the module docstring."""


class PALUTaskLogParser(PALUMixin, PALTaskLogParser):
    """PAL-U on System 1: the task's pal_events.jsonl."""
