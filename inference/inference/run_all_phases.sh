#!/bin/bash

set -eu

nsamples=50000
st_data_dir=../model/stretching/data
relax_data_dir=../model/relaxation/data
surrogate_dir=../surrogate/nn_surrogate/trained
cores=8

phase_1_dir=results_phase_1
phase_2_dir=results_phase_2
phase_3_thetanew_dir=results_phase_3_thetanew
phase_3_exp_dir=results_phase_3_exp

phase_1() {
    ./run_phase_1.py \
	$surrogate_dir \
	--nsamples $nsamples \
	--cores $cores \
	--stretch-exp $st_data_dir/mills_2004/mills_2004.csv $st_data_dir/suresh_2005/suresh_2005.csv \
	--relax-exp $relax_data_dir/hochmut_1979/*.csv \
	--output-dir $phase_1_dir
}

phase_2() {
    ./run_phase_2.py \
	$phase_1_dir \
	--nsamples $nsamples \
	--cores $cores \
	--output-dir $phase_2_dir
}

phase_3_thetanew() {
    ./run_phase_3_thetanew.py \
	$phase_2_dir \
	--nsamples $nsamples \
	--cores $cores \
	--output-dir $phase_3_thetanew_dir
}

phase_3_exp() {
    ./run_phase_3_exp.py \
	$surrogate_dir \
	$phase_1_dir \
	$phase_2_dir \
	--nsamples $nsamples \
	--cores $cores \
	--stretch-exp $st_data_dir/mills_2004/mills_2004.csv $st_data_dir/suresh_2005/suresh_2005.csv \
	--relax-exp $relax_data_dir/hochmut_1979/*.csv \
	--output-dir $phase_3_exp_dir
}

collect_samples() {
    samples_dir=samples
    mkdir -p samples

    ./korali_to_csv.py $phase_1_dir/eq/latest $samples_dir/posterior.single.eq.csv
    ./korali_to_csv.py $phase_1_dir/stretch_0/latest $samples_dir/posterior.single.st0.csv
    ./korali_to_csv.py $phase_1_dir/stretch_1/latest $samples_dir/posterior.single.st1.csv
    ./korali_to_csv.py $phase_1_dir/relax_0/latest $samples_dir/posterior.single.re0.csv
    ./korali_to_csv.py $phase_1_dir/relax_1/latest $samples_dir/posterior.single.re1.csv
    ./korali_to_csv.py $phase_1_dir/relax_2/latest $samples_dir/posterior.single.re2.csv
    ./korali_to_csv.py $phase_1_dir/relax_3/latest $samples_dir/posterior.single.re3.csv

    ./korali_to_csv.py $phase_2_dir/latest $samples_dir/posterior.hyperparams.csv

    ./korali_to_csv.py $phase_3_thetanew_dir/latest $samples_dir/posterior.thetanew.csv

    ./korali_to_csv.py $phase_3_exp_dir/eq/latest $samples_dir/posterior.hierarchical.eq.csv
    ./korali_to_csv.py $phase_3_exp_dir/stretch_0/latest $samples_dir/posterior.hierarchical.st0.csv
    ./korali_to_csv.py $phase_3_exp_dir/stretch_1/latest $samples_dir/posterior.hierarchical.st1.csv
    ./korali_to_csv.py $phase_3_exp_dir/relax_0/latest $samples_dir/posterior.hierarchical.re0.csv
    ./korali_to_csv.py $phase_3_exp_dir/relax_1/latest $samples_dir/posterior.hierarchical.re1.csv
    ./korali_to_csv.py $phase_3_exp_dir/relax_2/latest $samples_dir/posterior.hierarchical.re2.csv
    ./korali_to_csv.py $phase_3_exp_dir/relax_3/latest $samples_dir/posterior.hierarchical.re3.csv
}

phase_1
phase_2
phase_3_thetanew
phase_3_exp

collect_samples
