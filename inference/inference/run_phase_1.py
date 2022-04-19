#!/usr/bin/env python

import korali
import numpy as np
import os
import pandas as pd
from scipy.stats import norm
import sys

from common import exponential_model

here = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, os.path.join(here, ".."))

from prior import (VarConfig,
                   UniformPrior,
                   RatioUniformPrior,
                   comp_variables_dict)

sys.path.insert(0, os.path.join(here, "..", "surrogate"))
sys.path.insert(0, os.path.join(here, "..", "surrogate", 'nn_surrogate')) # beacause of pickle

from nn_surrogate import Surrogate

def run_experiment_equilibrium(k, *,
                               surrogate: Surrogate,
                               nsamples: int,
                               res_name: str,
                               D_ref:    float=7.82,
                               hmin_ref: float=0.81,
                               hmax_ref: float=2.58):

    def model(sample):
        v, FvK, sigma = sample['Parameters']

        mu = 4 # arbitrary, not sensitive
        b2 = 2 # arbitrary, not sensitive

        D, hmin, hmax = surrogate.evaluate_equilibrium([v, mu, FvK, b2])
        y = [D, hmin, hmax]

        sample["Reference Evaluations"] = y
        sample["Standard Deviation"] = [sigma * val for val in y]

    e = korali.Experiment()

    e["Problem"]["Type"] = "Bayesian/Reference"
    e["Problem"]["Likelihood Model"] = "Normal"
    e["Problem"]["Reference Data"] = [D_ref, hmin_ref, hmax_ref]
    e["Problem"]["Computational Model"] = model


    e["Solver"]["Type"] = "Sampler/TMCMC"
    e["Solver"]["Population Size"] = nsamples
    e["Solver"]["Target Coefficient Of Variation"] = 0.3
    e["Solver"]["Covariance Scaling"] = 0.08

    # e["Solver"]["Type"] = "Sampler/Nested"
    # e["Solver"]["Resampling Method"] = "Multi Ellipse"
    # e["Solver"]["Number Live Points"] = nsamples

    variables = [comp_variables_dict['v'],
                 comp_variables_dict['FvK'],
                 VarConfig(name="sigma",
                           prior=UniformPrior(a=0.0, b=0.5))]

    for i, var in enumerate(variables):
        var.configure_korali(e, i)

    e["File Output"]["Path"] = res_name
    e["Console Output"]["Verbosity"] = "Detailed"
    # e["Console Output"]["Frequency"] = 50
    # e["File Output"]["Frequency"] = 500
    k.run(e)



def run_experiment_stretching(k, *,
                              surrogate: Surrogate,
                              nsamples: int,
                              res_name: str,
                              csv_ref: str):

    df = pd.read_csv(csv_ref)

    Fext = df["Fext"].tolist()
    D0_ref = df["D0"].to_numpy()
    D1_ref = df["D1"].to_numpy()
    RA = np.sqrt(135 / (4 * np.pi))

    def model(sample):
        v, mu, kb, b2, sigma0, sigma1 = sample["Parameters"]
        FvK = mu * RA**2 / kb

        D0, D1 = surrogate.evaluate_stretching([v, mu, FvK, b2], Fext)
        D0 = np.array(D0)
        D1 = np.array(D1)

        sample["Reference Evaluations"] = D0.tolist() + D1.tolist()
        sample["Standard Deviation"] = [sigma0] * len(D0) + [sigma1] * len(D1)

    e = korali.Experiment()

    e["Problem"]["Type"] = "Bayesian/Reference"
    e["Problem"]["Likelihood Model"] = "Normal"
    e["Problem"]["Reference Data"] = D0_ref.tolist() + D1_ref.tolist()
    e["Problem"]["Computational Model"] = model


    e["Solver"]["Type"] = "Sampler/TMCMC"
    e["Solver"]["Population Size"] = nsamples
    e["Solver"]["Target Coefficient Of Variation"] = 0.4
    e["Solver"]["Covariance Scaling"] = 0.06

    # e["Solver"]["Type"] = "Sampler/Nested"
    # e["Solver"]["Resampling Method"] = "Multi Ellipse"
    # e["Solver"]["Number Live Points"] = nsamples

    variables = [comp_variables_dict['v'],
                 comp_variables_dict['mu'],
                 comp_variables_dict['kb'],
                 comp_variables_dict['b2'],
                 VarConfig(name="sigma0",
                           prior=UniformPrior(a=0.0, b=2.0)),
                 VarConfig(name="sigma1",
                           prior=UniformPrior(a=0.0, b=2.0))]


    for i, var in enumerate(variables):
        var.configure_korali(e, i)

    e["File Output"]["Path"] = res_name
    e["Console Output"]["Verbosity"] = "Detailed"
    # e["Console Output"]["Frequency"] = 50
    # e["File Output"]["Frequency"] = 500
    k.run(e)


def run_experiment_relaxation(k, *,
                              surrogate: Surrogate,
                              nsamples: int,
                              res_name: str,
                              relax_exp_file: str):
    """
    Infer all parameters of the generative model
    (etam/mu, sigma_tc) -> (tc, z0, zinf, sigma_z) -> y
    at once.
    """

    df = pd.read_csv(relax_exp_file)

    t = df["t"].to_numpy()
    y = df["L_W"].to_numpy()

    def model(ks):
        etam_mu, sigma_tc, tc, z0, zinf, sigma_z = ks["Parameters"]
        v = 0.95
        mu = 4
        FvK = 250
        b2 = 1.6
        etam = etam_mu * mu

        # use the same form as in Hochmut 1979
        z = exponential_model(z0, zinf, tc, t)

        log_likelihood = np.sum(norm.logpdf(y, loc=z, scale=sigma_z))

        tc_sim = surrogate.evaluate_relax([v, mu, FvK, b2, etam])

        log_likelihood += norm.logpdf(tc, loc=tc_sim, scale=tc*sigma_tc)

        ks["logLikelihood"] = log_likelihood

    e = korali.Experiment()

    e["Problem"]["Type"] = "Bayesian/Custom"
    e["Problem"]["Likelihood Model"] = model

    e["Solver"]["Type"] = "Sampler/TMCMC"
    e["Solver"]["Population Size"] = nsamples
    e["Solver"]["Target Coefficient Of Variation"] = 0.4
    e["Solver"]["Covariance Scaling"] = 0.06

    conf_mu = comp_variables_dict['mu']
    conf_etam = comp_variables_dict['etam']

    variables = [VarConfig(name="etam_mu",
                           prior=RatioUniformPrior(a=conf_etam.low(), b=conf_etam.high(),
                                                   c=conf_mu.low(), d=conf_mu.high())),
                 VarConfig(name="sigma_tc",
                           prior=UniformPrior(a=0.0, b=0.05)),
                 VarConfig(name="tc",
                           prior=UniformPrior(a=0.05, b=0.5)),
                 VarConfig(name="z0",
                           prior=UniformPrior(a=y[0]-0.2, b=y[0]+0.2)),
                 VarConfig(name="zinf",
                           prior=UniformPrior(a=0.8, b=1.5)),
                 VarConfig(name="sigma_z",
                           prior=UniformPrior(a=0.0, b=0.5))]


    for i, var in enumerate(variables):
        var.configure_korali(e, i)

    e["File Output"]["Path"] = res_name
    e["Console Output"]["Verbosity"] = "Detailed"
    k.run(e)



def main(argv):
    import argparse
    parser = argparse.ArgumentParser(description='Compute posterior distributions for all experiments using the NN surrogate.')
    parser.add_argument('base_dir', type=str, help="The directory that contains the trained states of the NN.")
    parser.add_argument('--nsamples', type=int, default=2000, help="Number of samples.")
    parser.add_argument('--stretch-exp', type=str, nargs='+', default=[], help="Stretching experiment files.")
    parser.add_argument('--relax-exp', type=str, nargs='+', default=[], help="Relaxation experiment files.")
    parser.add_argument('--output-dir', type=str, default="results_phase_1", help="The directory that will contain the results.")
    parser.add_argument('--cores', type=int, default=4, help="Number of cores to use.")
    args = parser.parse_args(argv)

    surrogate = Surrogate(args.base_dir)
    experiments = []
    output_dir = args.output_dir

    k = korali.Engine()
    k["Conduit"]["Type"] = "Concurrent"
    k["Conduit"]["Concurrent Jobs"] = args.cores

    run_experiment_equilibrium(k,
                               surrogate=surrogate,
                               nsamples=args.nsamples,
                               res_name=os.path.join(output_dir, "eq"))

    for i, f in enumerate(args.stretch_exp):
        run_experiment_stretching(k,
                                  surrogate=surrogate,
                                  nsamples=args.nsamples,
                                  res_name=os.path.join(output_dir, f"stretch_{i}"),
                                  csv_ref=f)

    for i, f in enumerate(args.relax_exp):
        run_experiment_relaxation(k,
                                  surrogate=surrogate,
                                  relax_exp_file=f,
                                  nsamples=args.nsamples,
                                  res_name=os.path.join(output_dir, f"relax_{i}"))


if __name__ == "__main__":
    main(sys.argv[1:])
