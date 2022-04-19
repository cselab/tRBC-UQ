#!/bin/bash

set -eu

R="3.3_um"
RA=12
NX=1

. mir.load # make sure we do not use an old python

launch_one_case()
{
    pg=$1; shift
    Vmax=`python -c "print($pg * 1/45)"` #cm/s

    pg="${pg}_mmHg_per_mm"
    Vmax="${Vmax}_cm_per_s"
    echo "pg = $pg, Vmax = $Vmax"
    ./run.daint.sh --pg=$pg --Vmax=$Vmax \
		   --R=$R --RA=$RA --NX=$NX \
		   --case="rbc_tube_pg_R_${R}_RA_${RA}" \
		   --num-restarts=3
}

for pg in `seq 5 5 75`; do
    launch_one_case $pg
done
