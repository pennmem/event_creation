#!/usr/bin/env bash
#$ -q RAM.q
#$ -cwd
#$ -N regression_events
source ~/.bashrc
cd ~/event_creation
rm *.png
source activate event_creation

python -m tests.regression_tests --db-root=/scratch/db_root $@