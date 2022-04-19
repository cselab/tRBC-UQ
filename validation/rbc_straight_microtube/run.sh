#!/bin/bash

set -eu

# default arguments
L="50_um"
R="3.3_um"
pg="45_mmHg_per_mm"
Vmax="1_cm_per_s"
etao="1e-3_Pa_s"
scale_ini=0.5
Re=0.1
RA=6
C=1 # this should not matter at the steady state; use 1 for lower simulation time
rbc_res=4
dry_run=false
membrane_visc=true
extra_args=""

NX=1
NY=1
NZ=1

SCRIPTPATH="$( cd "$(dirname "$0")" >/dev/null 2>&1 ; pwd -P )"

usage()
{
    cat <<EOF
usage: $0
       [-h | --help] Print this help message and exit
       [--dry-run] Print the parameters and exit
       [--no-visc] Disable membrane viscosity
       [--pg=<pg>] Set the pressure gradient (default: $pg)
       [--C=<C>] Set the viscosity ratio between inner and outer solvent (default: $C)
       [--Re=<Re>] Set the simulation Reynolds number (default: $Re)
       [--RA=<RA>] Set the equivalent radius of the RBC in simulation units (default: $RA)
       [--rbc_res=<rbc_res>] Number of subdivisions of the RBC mesh (default: $rbc_res)
       [--Vmax=<Vmax>] Set the mean blood velocity (default: $Vmax)
       [--etao=<etao>] Set the viscosity of the outer fluid (default: $etao)
       [--scale_ini=<scale_ini>] Set the initial RBCs scale (default: $scale_ini)
       [--R=<R>] The pipe radius (default: $R)
       [--L=<L>] The pipe length (default: $L)
       [--NX=<NX>] number of ranks in the X direction (default: $NX)
       [--NY=<NY>] number of ranks in the Y direction (default: $NY)
       [--NZ=<NZ>] number of ranks in the Z direction (default: $NZ)
EOF
}

# parse optional arguments
while test $# -ne 0; do
    case "$1" in
	--NX=*)        NX="${1#*=}"        ; shift ;;
	--NY=*)        NY="${1#*=}"        ; shift ;;
	--NZ=*)        NZ="${1#*=}"        ; shift ;;
	--dry-run)     dry_run=true        ; shift ;;
	--no-visc)     membrane_visc=false ; shift ;;
	--pg=*)        pg="${1#*=}"        ; shift ;;
	--C=*)         C="${1#*=}"         ; shift ;;
	--Re=*)        Re="${1#*=}"        ; shift ;;
	--RA=*)        RA="${1#*=}"        ; shift ;;
	--rbc_res=*)   rbc_res="${1#*=}"   ; shift ;;
	--Vmax=*)      Vmax="${1#*=}"      ; shift ;;
	--etao=*)      etao="${1#*=}"      ; shift ;;
	--scale_ini=*) scale_ini="${1#*=}" ; shift ;;
	--L=*)         L="${1#*=}"         ; shift ;;
	--R=*)         R="${1#*=}"         ; shift ;;
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

srcdir=$SCRIPTPATH
rundir=R_${R}_pg_${pg}_Vmax_${Vmax}_etao_${etao}

if [ $membrane_visc = false ]; then
    rundir=${rundir}_no_visc
    extra_args="${extra_args} --no-visc"
fi


parameters="settings.pkl"

mkdir -p $rundir
cp $srcdir/parameters.py $rundir
cp $srcdir/generate_ic.py $rundir
cp $srcdir/main.py $rundir
cd $rundir

NRANKS=`python -c "print(2 * $NX * $NY * $NZ)"`

params()
{
    mir.run -n 1 ./parameters.py \
	    --rbc-res $rbc_res \
	    --L $L --R $R \
	    --Vmax $Vmax \
	    --pressure-gradient $pg \
	    --Re $Re \
	    --RA $RA \
	    --eta-out $etao \
	    --C $C \
	    --out-params $parameters
}

ic()
{
    mir.run -n $NRANKS ./generate_ic.py \
	    --ranks $NX $NY $NZ \
	    --params $parameters \
	    --scale-ini $scale_ini
}

run()
{
    mir.run -n $NRANKS ./main.py \
	    --ranks $NX $NY $NZ \
	    --params $parameters \
	    $extra_args \
	    $@
}

params

if [ $dry_run = true ]; then
    echo "dry-run enabled, exiting."
    exit 0
fi

ic
run

# example for restart
# run --checkpoint-dir "tmp0"
# run --restart-dir "tmp0" --checkpoint-dir "tmp1"
