#!/bin/bash

set -eu

ns=32

for name in 'v' 'mu' 'FvK' 'b2' 'etam' 'Fext'; do
    param_space=lines/along_${name}_${ns}.csv
    ./lines/generate.py --along $name --nsamples $ns --out $param_space
    ./run_samples.local.sh $param_space
    mv samples.csv lines_evaluations/along_${name}_${ns}.csv
done
