#!/bin/bash

set -eu

sample=$1; shift

SCRIPTPATH="$( cd "$(dirname "$0")" >/dev/null 2>&1 ; pwd -P )"

mesh_dir=$SCRIPTPATH/../../model/mesh

mir.run -n 4 ./run_samples.py \
	--sample-list $sample \
	--mesh-sphere $mesh_dir/sphere/sph_L4.off \
	--mesh-ini $mesh_dir/rbc_lwm_minimum.off
