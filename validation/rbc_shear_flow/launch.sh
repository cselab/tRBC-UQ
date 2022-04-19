#/bin/bash

set -eu

RA=8
Re=0.14

launch_case() (
    case_name=$1; shift
    etao=$1; shift

    low_sh=$1; shift
    high_sh=$1; shift
    n=$1; shift

    # generate sequence in logarithmic scale
    seq_shear_rates=`python <<EOF
import numpy as np
lo=$low_sh
hi=$high_sh
for val in np.logspace(np.log(lo)/np.log(10), np.log(hi)/np.log(10), $n):
    print(val)
EOF`

    for shear in $seq_shear_rates; do
	shear_rate="${shear}_Hz"
	echo $shear_rate
        ./run.daint.sh \
	    --Re=$Re \
	    --RA=$RA \
	    --etao=$etao \
	    --etai="10_mPa_s" \
	    --shear-rate=$shear_rate \
	    --case-name=$case_name
    done
)

# for angle data
#launch_case rbc_shear_RA_${RA}_eta_10.7_mPas "10.7_mPa_s" 1.0 200.0 14
#launch_case rbc_shear_RA_${RA}_eta_23.9_mPas "23.9_mPa_s" 1.0 200.0 14
## launch_case rbc_shear_RA_${RA}_eta_55.9_mPas "55.9_mPa_s" 1.0 200.0 14
## launch_case rbc_shear_RA_${RA}_eta_104_mPas "104_mPa_s" 1.0 200.0 14

# for TTF data
#launch_case rbc_shear_RA_${RA}_eta_28.9_mPas "28.9_mPa_s" 8.0 264.0 6

# for critical angles
#launch_case rbc_shear_threshold_RA_${RA}_eta_10_mPas "10_mPa_s" 30 300.0 5
launch_case rbc_shear_threshold_RA_${RA}_eta_14_mPas "14_mPa_s" 7 70 7
#launch_case rbc_shear_threshold_RA_${RA}_eta_20_mPas "20_mPa_s" 5 50.0 5
launch_case rbc_shear_threshold_RA_${RA}_eta_23.9_mPas "23.9_mPa_s" 1 12 7
#launch_case rbc_shear_threshold_RA_${RA}_eta_30_mPas "30_mPa_s" 1 4.0 5
#launch_case rbc_shear_threshold_RA_${RA}_eta_50_mPas "50_mPa_s" 1 4 5
launch_case rbc_shear_threshold_RA_${RA}_eta_70_mPas "70_mPa_s" 0.5 2 7
#launch_case rbc_shear_threshold_RA_${RA}_eta_100_mPas "100_mPa_s" 0.3 1.0 5
