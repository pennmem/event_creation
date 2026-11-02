# PAL1 / IPAL1 on System 1 (sync box, no host PC)

How to process a PAL1 or IPAL1 session that was run with the UnityEPL task
(pennmem PAL1_UnityEPL) and a Penn sync box recording into the clinical EEG, as at
Colorado (CU Anschutz, subject suffix A, Nihon Kohden). For System 4 (Elemem) sessions
nothing here applies; see `parsers/elemem_pal_parser.py`.

## What the pipeline uses

| Piece | Where |
|---|---|
| Events | `parsers/pal_tasklog_parser.py` (`PALTaskLogParser`) reads the task's `pal_events.jsonl` and builds the same events and fields as the System 4 parser (`parsers/pal_base.py` holds the shared scoring) |
| PAL1 math | `parsers/math_parser.py` (`MathPALTaskLogParser`), from the same file |
| Sync pulses on the EEG | `readers/sync_pulse_extractor.py`, run by hand before the import (below) |
| Alignment | `alignment/system1_tasklog.py` (`TaskLogSystem1Aligner`): the lab's `System1Aligner`, with the task's pulse times read from `pal_events.jsonl` on the clock its events are on |
| Routing | `--sys1` puts `system_1` in the groups; it overrides the subject-suffix rule (A would otherwise be System 3.3). `pipelines.determine_groups` adds `pal_task_log` for IPAL, and for PAL when the session has `pal_events.jsonl` |
| Transfer | `transfer_inputs/behavioral_inputs.yml`, entries with groups `[pal_task_log, system_1]`; the sync pulses are the standard System 1 entry (`find_sync_file`) |

## Layout

Under `data_root` (`/data/eeg` on rhino):

```
R1765A/
  behavioral/IPAL1/session_0/
      pal_events.jsonl            required: the task log (LAUNCH, syncPulse, TRIAL, ... lines)
      0_0.ann ... 25_5.ann        required: one per probe, <list>_<slot>.ann (practice is list 0)
      0_0.wav ... 25_5.wav        optional, copied with the annotations
      *.lst                       optional, archived
      (interrupted/, session.jsonl and the rest are not read)
  raw/IPAL1_0/
      <recording>.edf             the clinical EEG, exported to EDF (preferred), or
      <recording>.EEG + .21E      a Nihon Kohden recording (reader untested on real files)
      sync_channel.json           which input(s) carry the pulses (below)
      <recording>.sync.txt        written by the extractor
```

The folder names use the experiment and the task's session number
(`raw/<experiment>_<original_session>`). Copy the session folder from the task laptop
(`data/IPAL1/<subject>/session_N/`) to `behavioral/IPAL1/session_N/`.

A montage (`contacts.json` for localization/montage 0.0) must exist first, as for any
System 1 import; the split keeps only the channels it lists, so the sync inputs need not
be in it.

### `sync_channel.json`

The two inputs the box's + and - leads went into, as written in the session notes
(the extractor takes the first minus the second):

```json
{"channels": ["C129", "C130"]}
```

or one input (a DC channel, or an EKG input):

```json
{"channel": "DC01"}
```

Optional keys: `"polarity": "auto" | "positive" | "negative"` (default auto) and
`"threshold"` (physical units above the baseline; default half the pulse height). Labels
match loosely: case, spaces, an `EEG ` prefix, a `-Ref` suffix and leading zeros are
ignored, so `C129` finds `EEG C129-Ref`. EKG-labelled inputs work here even though the
split drops them.

## Commands

```
R=/path/to/rhino_root          # on rhino: /
cd <event_creation checkout>
export PYTHONPATH=$PWD

# 1. Find the pulses: writes raw/IPAL1_0/<recording>.sync.txt for every .edf/.EEG there
python -m event_creation.submission.readers.sync_pulse_extractor $R/data/eeg/R1765A/raw/IPAL1_0

# 2. Import as System 1
python -m event_creation.submission.convenience \
  --path "rhino_root=${R}:data_root=${R}/data/eeg:db_root=${R}:events_root=${R}/data/events:loc_db_root=${R}/home2/RAM_maint/stim" \
  --set-input protocol=r1:code=R1765A:subject=R1765A:experiment=IPAL1:session=0:original_session=0:montage=0.0 \
  --sys1
```

On rhino, `./submit --sys1 --set-input ...` does the same (no `--path`). In zsh write
`${R}` with braces: `$R:e` is a zsh modifier. For a JSON import (`--json`), give the
session `"system_1": true`. Add `--force-events --force-eeg` to redo an import.

The extractor prints, per file, the number of pulses, the polarity, threshold, pulse width
and the interval range; check that the width is about 20 ms and intervals 800-1200 ms
(except gaps between launches). Its indices are at the rate the split writes to
`sources.json`: an EDF at 10 kHz or more is split at 1 kHz, and the extractor converts.

The aligner logs its fit (slope, intercept, residuals) at debug level only.

## What the alignment does and refuses

- The task's stopwatch restarts at each launch (a resumed session has several
  `LAUNCH` lines). Event `mstime` is `LAUNCH.epochUnixMs + time`, and the `syncPulse`
  times are read on the same clock, all launches together. Everything after that is
  `System1Aligner`, as for every other System 1 experiment.
- It matches a window of pulse intervals (100 pulses, shrinking by 10 to 5) at the start
  and at the end of the session, each interval within 10 ms, and fits one line through
  the matched pulses. Missing pulses between the two windows, pre-session "Test Syncbox"
  pulses and a recording that starts after the task are fine. There are no checks on the
  slope or the residuals of the fit.
- `AlignmentError('Could not align window')` if no window of 5 or more matches at either
  end; `AlignmentError('Could not align any events.')` if every event falls outside the
  recording.
- Events outside the recording get `eegoffset` -1 and an empty `eegfile`.
- One EEG file only: `AlignmentError('Cannot align EEG with N sources')` if
  `sources.json` lists more than one, and `find_sync_file` needs exactly one `.sync.txt`
  in `raw/<experiment>_<session>/`. If the clinical system split the recording
  (Nihon Kohden splits at 90 min / 6 h), export the session as one EDF.

## Not covered

- The delay between the task asking for a pulse and its edge on the EEG (helper +
  LabJack USB) is a constant bias the fit cannot see; measure it on the bench.
- The NK `.EEG` path goes through the existing `NK_reader` and has not been run on a real
  file; export EDF when possible.
- PAL1 math events keep every attempt's problems after a restart (as on System 4).
