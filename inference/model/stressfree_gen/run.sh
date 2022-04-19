#!/bin/bash

set -eu

rV=0.96
out=final.ply

usage()
{
    cat <<EOF
usage: $0
       [-h | --help] Print this error message
       [--rV=<rV>] The desired reduced volume (default: $rV)
       [--out=<out>] The result mesh name (default: $out)
EOF
}

# parse optional arguments
while test $# -ne 0; do
    case "$1" in
	-h|--help) usage; exit 0 ;;
	--rV=*)  rV="${1#*=}" ; shift ;;
	--out=*) out="${1#*=}"; shift ;;
	-*|--*) echo "Error: unsupported option $1"            ; usage; exit 1 ;;
	*)      echo "Error: no positional arguments required."; usage; exit 1 ;;
    esac
done



SCRIPTPATH="$( cd "$(dirname "$0")" >/dev/null 2>&1 ; pwd -P )"

mesh_dir=$SCRIPTPATH/../mesh/sphere

mir.run -n 1 ./main.py \
	--mesh_ini $SCRIPTPATH/ref_v0.95.off \
	--mesh_ref $mesh_dir/sph_L4.off \
	--reduced-volume $rV \
	--out $out
