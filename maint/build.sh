#!/usr/bin/env bash

EVENT_CREATION_ROOT=`dirname $0`/../
export DEPENDENCIES=`sed -ne '/^dependencies:$/{s///; :a' -e 'n;p;ba' -e '}' ${EVENT_CREATION_ROOT}/conda_environment.yml`
conda build -c pennmem --output-folder  "${EVENT_CREATION_ROOT}/build/" "${EVENT_CREATION_ROOT}/conda.recipe"