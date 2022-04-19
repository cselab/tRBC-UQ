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
membrane_visc=true
extra_args=""
case_name="rbc_tube"
num_restarts=0

NX=1
NY=1
NZ=1

SCRIPTPATH="$( cd "$(dirname "$0")" >/dev/null 2>&1 ; pwd -P )"

usage()
{
    cat <<EOF
usage: $0
       [-h | --help] Print this help message and exit
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
       [--case=<case_name>] The name of the directory containing the simulation output directory (default: $case_name).
       [--num-restarts=<num_restarts>] The number of additional jobs of 24 hours to launch (due to time limit in slurm) (default: $num_restarts).
       [--NX=<NX>] number of ranks in the X direction (default: $NX)
       [--NY=<NY>] number of ranks in the Y direction (default: $NY)
       [--NZ=<NZ>] number of ranks in the Z direction (default: $NZ)
EOF
}

# parse optional arguments
while test $# -ne 0; do
    case "$1" in
	--NX=*)            NX="${1#*=}"        ; shift ;;
	--NY=*)            NY="${1#*=}"        ; shift ;;
	--NZ=*)            NZ="${1#*=}"        ; shift ;;
	--pg=*)            pg="${1#*=}"        ; shift ;;
	--no-visc)         membrane_visc=false ; shift ;;
	--C=*)             C="${1#*=}"         ; shift ;;
	--Re=*)            Re="${1#*=}"        ; shift ;;
	--RA=*)            RA="${1#*=}"        ; shift ;;
	--rbc_res=*)       rbc_res="${1#*=}"   ; shift ;;
	--Vmax=*)          Vmax="${1#*=}"      ; shift ;;
	--etao=*)          etao="${1#*=}"      ; shift ;;
	--scale_ini=*)     scale_ini="${1#*=}" ; shift ;;
	--L=*)             L="${1#*=}"         ; shift ;;
	--R=*)             R="${1#*=}"         ; shift ;;
	--case=*)          case_name="${1#*=}" ; shift ;;
	--num-restarts=*)  num_restarts="${1#*=}" ; shift ;;
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
rundir=$SCRATCH/$case_name/R_${R}_pg_${pg}_Vmax_${Vmax}_etao_${etao}_RA_${RA}

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

b0="sbatch0.sh"

num_nodes=`python -c "print($NX * $NY * $NZ)"`
num_ranks=`python -c "print(2 * $num_nodes)"`

checkpoint="checkpoint0"

cat > $b0 <<EOF
#!/bin/bash -l
#SBATCH --job-name=pg_${pg}_R_${R}
#SBATCH --time=24:00:00
#SBATCH --nodes=$num_nodes
#SBATCH --partition=normal
#SBATCH --constraint=gpu

. mir.load

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

mir.run -n $num_ranks ./generate_ic.py \
	--ranks $NX $NY $NZ \
	--params $parameters \
	--scale-ini $scale_ini

mir.run -n $num_ranks ./main.py \
	--ranks $NX $NY $NZ \
	--params $parameters \
	--checkpoint-dir $checkpoint \
	$extra_args

EOF

jobid=`sbatch $b0 | awk '{print $NF}'`
echo "Launched job $jobid"

for i in `seq $num_restarts`; do
    bnext="sbatch${i}.sh"

    restart=$checkpoint
    checkpoint="checkpoint${i}"

    cat > $bnext <<EOF
#!/bin/bash -l
#SBATCH --job-name=${i}_pg_${pg}_R_${R}
#SBATCH --time=24:00:00
#SBATCH --nodes=$num_nodes
#SBATCH --partition=normal
#SBATCH --constraint=gpu
#SBATCH --dependency=afterany:$jobid

. mir.load

mir.run -n $num_ranks ./main.py \
	--ranks $NX $NY $NZ \
	--params $parameters \
	--restart-dir $restart \
	--checkpoint-dir $checkpoint \
	$extra_args

EOF

    jobid=`sbatch $bnext | awk '{print $NF}'`
    echo "Launched job $jobid"
done
