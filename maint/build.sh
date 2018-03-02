#!/usr/bin/env bash

EVENT_CREATION_ROOT=`dirname $0`/../
conda build -c pennmem --output-folder  "${EVENT_CREATION_ROOT}/build/" "${EVENT_CREATION_ROOT}/conda.recipe"