#!/bin/bash

set -eu

SCRIPTPATH="$( cd "$(dirname "$0")" >/dev/null 2>&1 ; pwd -P )"

kind=$1; shift
ns=$1; shift

mesh_dir=$SCRIPTPATH/../../model/mesh
sample_dir=$SCRIPTPATH/$kind

samples=samples-${ns}.csv

sbatch=sbatch_${kind}_${ns}.sh
num_nodes=256

res_dir=${kind}_evaluations

mkdir -p $res_dir

cat > $sbatch <<EOF
#!/bin/bash -l
#SBATCH --job-name=eval_${ns}
#SBATCH --time=24:00:00
#SBATCH --nodes=$num_nodes
#SBATCH --partition=normal
#SBATCH --constraint=gpu

. mir.load

mir.run -n $num_nodes ./run_samples.py \
	--sample-list $sample_dir/$samples \
	--mesh-sphere $mesh_dir/sphere/sph_L4.off \
	--mesh-ini $mesh_dir/rbc_lwm_minimum.off \
	--out $SCRIPTPATH/$res_dir/samples_${ns}.csv
EOF

sbatch $sbatch
