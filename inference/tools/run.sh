#!/bin/bash

set -eu

mesh_dir=../model/mesh
res_dir=results

mkdir -p $res_dir

make_mesh() {
    v=$1; shift
    L=$1; shift
    ext=$1; shift
    mir.run -n 1 ./create_rbc_meshes.py \
	--reduced-volume $v \
	--mesh-sphere $mesh_dir/sphere/sph_L${L}.off \
	--out-sf-mesh $res_dir/S0_v_${v}_L_${L}.$ext \
	--out-eq-mesh $res_dir/eq_v_${v}_L_${L}.$ext
}

for v in 0.93 0.94 0.95 0.96 0.97; do
    for L in 3 4; do
	make_mesh $v $L off
    done
done

exit 0

L=4
for v in 0.65 0.70 0.75 0.80 0.85 0.90 0.95 0.99; do
    make_mesh $v $L ply
done
