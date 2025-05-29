#!/bin/bash

source ~/.bashrc

SCRIPTDIR="$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")" >/dev/null 2>&1 && pwd)"
cd "$SCRIPTDIR"

LOG=~/logs/automation_log.txt
mkdir ~/logs 2>/dev/null
mkdir ~/logs/event_creation_outputs 2>/dev/null
mkdir ~/logs/stdouterr 2>/dev/null

# Automatically run event creation for recently modified sessions that have yet
# to be processed
echo "$(date): Start." &>>"$LOG"
./automatic_pipeline_ltp.py &>>"$LOG" &&
echo "$(date): Success." &>>"$LOG" ||
echo "$(date): Error!" &>>"$LOG"

