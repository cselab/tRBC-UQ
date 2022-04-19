#!/bin/bash

set -eu

R="3.3_um"

launch_one_case()
{
    RA=$1; shift
    echo "RA=$RA"
    ./run.daint.sh --case="tbc_tube_resolutions" --R=$R --RA=$RA
}

for RA in 4 5 6 7 8 9 10 11 12; do
    launch_one_case $RA
done
