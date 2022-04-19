#!/bin/bash

set -eu

R="3.35_um"
RA=12
NX=1

. mir.load # make sure we do not use an old python

launch_one_case()
{
    v=$1; shift
    pg=`python -c "print($v * 45)"`

    pg="${pg}_mmHg_per_mm"
    Vmax="${v}_cm_per_s"
    echo "pg = $pg, Vmax = $Vmax"
    ./run.daint.sh --pg=$pg --Vmax=$Vmax \
		   --R=$R --RA=$RA --NX=$NX \
		   --case="rbc_tube_v_l_R_${R}_RA_${RA}" \
		   --num-restarts=3
}



low_v=0.03 # cm/s
high_v=2
n_v=8


velocities=`python <<EOF
import numpy as np
lo=$low_v
hi=$high_v
for val in np.logspace(np.log(lo)/np.log(10), np.log(hi)/np.log(10), $n_v):
    print(val)
EOF`


for v in $velocities; do
    launch_one_case $v
done
