#!/bin/bash

set -eu

samples_dir=../inference/samples
surrogate_dir=../surrogate/nn_surrogate/trained

results_dir=results
mkdir -p $results_dir

single_level() {
    ./run_equilibrium.py \
	$surrogate_dir \
	$samples_dir/posterior.single.eq.csv \
	--output-csv $results_dir/forward.single.eq.csv

    ./run_stretching.py \
	$surrogate_dir \
	$samples_dir/posterior.single.st0.csv \
	--output-csv $results_dir/forward.single.st0.csv

    ./run_stretching.py \
	$surrogate_dir \
	$samples_dir/posterior.single.st1.csv \
	--output-csv $results_dir/forward.single.st1.csv

    ./run_relaxation.py \
	$samples_dir/posterior.single.re0.csv \
	--output-csv $results_dir/forward.single.re0.csv
    ./run_relaxation.py \
	$samples_dir/posterior.single.re1.csv \
	--output-csv $results_dir/forward.single.re1.csv
    ./run_relaxation.py \
	$samples_dir/posterior.single.re2.csv \
	--output-csv $results_dir/forward.single.re2.csv
    ./run_relaxation.py \
	$samples_dir/posterior.single.re3.csv \
	--output-csv $results_dir/forward.single.re3.csv
}

hierarchical() {
    ./run_equilibrium.py \
	$surrogate_dir \
	$samples_dir/posterior.hierarchical.eq.csv \
	--output-csv $results_dir/forward.hierarchical.eq.csv

    ./run_stretching.py \
	$surrogate_dir \
	$samples_dir/posterior.hierarchical.st0.csv \
	--output-csv $results_dir/forward.hierarchical.st0.csv

    ./run_stretching.py \
	$surrogate_dir \
	$samples_dir/posterior.hierarchical.st1.csv \
	--output-csv $results_dir/forward.hierarchical.st1.csv

    ./run_relaxation.py \
	$samples_dir/posterior.hierarchical.re0.csv \
	--output-csv $results_dir/forward.hierarchical.re0.csv
    ./run_relaxation.py \
	$samples_dir/posterior.hierarchical.re1.csv \
	--output-csv $results_dir/forward.hierarchical.re1.csv
    ./run_relaxation.py \
	$samples_dir/posterior.hierarchical.re2.csv \
	--output-csv $results_dir/forward.hierarchical.re2.csv
    ./run_relaxation.py \
	$samples_dir/posterior.hierarchical.re3.csv \
	--output-csv $results_dir/forward.hierarchical.re3.csv
}

single_level
hierarchical
