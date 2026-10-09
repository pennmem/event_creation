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
| Alignment | `alignment/system1_tasklog.py` (`TaskLogSystem1Aligner`): one fit per task launch and EEG file, with slope, residual and matched-fraction gates |
| Routing | `--sys1` puts `system_1` in the groups; it overrides the subject-suffix rule (A would otherwise be System 3.3). `pipelines.determine_groups` adds `pal_task_log` for IPAL, and for PAL when the session has `pal_events.jsonl` |
| Transfer | `transfer_inputs/behavioral_inputs.yml`, entries with groups `[pal_task_log, system_1]` |

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
      <recording 2>.edf           a second file if the recording was split (NK splits at 90 min / 6 h)
      sync_channel.json           which input(s) carry the pulses (below)
      <recording>.sync.txt        one per recording, written by the extractor
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

The import logs one line per fit, e.g.

```
Sync fit launch 2, R1765A_IPAL1_0_part2.edf: 117 of the 117 task pulses inside the file matched (174 in the launch), slope +37.8 ppm, RMS 0.62 ms, max 2.04 ms, 0 outliers dropped
```

## What the alignment does and refuses

- The task's stopwatch restarts at each launch (a resumed session has several
  `LAUNCH` lines). Event `mstime` is `LAUNCH.epochUnixMs + time`; each launch's
  `syncPulse` lines are fitted separately against the EEG pulses of each file.
- Pulses are matched on their intervals (median difference over ~20 intervals), so
  missing EEG pulses, pre-session "Test Syncbox" pulses and a recording that starts
  after the task are fine.
- `AlignmentError` if a fit's slope is more than 1 % from 1 (a sample-rate mismatch, not
  drift), if more than 1 % of matched pulses are over 5 ms off the line, or if fewer than
  90 % of the task pulses inside a file's pulse span match.
- Events outside every EEG file (or in a launch whose pulses match no file) get
  `eegoffset` -1.
- With several EEG files, each `.sync.txt` is paired with the file of the same stem.
  Two files whose EDF headers start in the same minute get the same split name and
  overwrite each other: do not set every export's start time to one fixed value when
  anonymising (shift them, or keep their order).

## Not covered

- The delay between the task asking for a pulse and its edge on the EEG (helper +
  LabJack USB) is a constant bias the fit cannot see; measure it on the bench.
- The NK `.EEG` path goes through the existing `NK_reader` and has not been run on a real
  file; export EDF when possible.
- PAL1 math events keep every attempt's problems after a restart (as on System 4).
