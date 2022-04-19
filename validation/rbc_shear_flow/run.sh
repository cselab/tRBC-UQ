#!/bin/bash

set -eu

# default parameters
etao="55.9_mPa_s"
shear_rate="50.0_Hz"
Re=0.14
RA=6

NX=1
NY=1
NZ=1

SCRIPTPATH="$( cd "$(dirname "$0")" >/dev/null 2>&1 ; pwd -P )"

usage()
{
    cat <<EOF
usage: $0
       [-h | --help] Print this help message and exit
       [--Re=<Re>] Set the simulation Reynolds number (default: $Re)
       [--RA=<RA>] Set the simulation simulation RBC equivalent radius (default: $RA)
       [--etao=<etao>] Set the viscosity of the solvent (default: $etao)
       [--shear-rate=<shear_rate>] Set the shear rate of the flow (default: $shear_rate)
       [--NX=<NX>] Number of ranks in the x direction (default: $NX)
       [--NY=<NY>] Number of ranks in the y direction (default: $NY)
       [--NZ=<NZ>] Number of ranks in the z direction (default: $NZ)
EOF
}

# parse optional arguments
while test $# -ne 0; do
    case "$1" in
	--Re=*)            Re="${1#*=}"            ; shift ;;
	--RA=*)            RA="${1#*=}"            ; shift ;;
	--etao=*)          etao="${1#*=}"          ; shift ;;
	--shear-rate=*)    shear_rate="${1#*=}"    ; shift ;;
	--NX=*)            NX="${1#*=}"            ; shift ;;
	--NY=*)            NY="${1#*=}"            ; shift ;;
	--NZ=*)            NZ="${1#*=}"            ; shift ;;
	-h|--help)
	    usage; exit 0 ;;
	-*|--*)
	    echo "Error: unsupported option $1"
	    usage; exit 1 ;;
	*)
	    echo "Error: no positional arguments required."
	    usage; exit 1 ;;
    esac
done


params="parameters.pkl"

src_dir=$SCRIPTPATH
run_dir=RA_${RA}_etao_${etao}_shear_rate_${shear_rate}

mkdir -p $run_dir

cp $src_dir/parameters.py $run_dir
cp $src_dir/main.py $run_dir
cd $run_dir

mir.run -n 1 ./parameters.py \
	--Re $Re \
	--RA $RA \
	--eta-solvent $etao \
	--shear-rate $shear_rate \
	--out-params $params

NRANKS=`python -c "print(2 * $NX * $NY * $NZ)"`

mir.run -n $NRANKS ./main.py \
	$params \
	--ranks $NX $NY $NZ \
	--dump
