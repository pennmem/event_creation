#!/usr/bin/env bash
source /etc/profile.d/modules.sh
module load SGE


qsub ~/event_creation/run_regression_test.sh