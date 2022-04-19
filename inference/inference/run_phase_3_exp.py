#!/usr/bin/env python

import korali
import numpy as np
import os
import pandas as pd
from scipy.stats import norm
import sys

from common import (exponential_model,
                    pdf_norm,
                    pdf_truncnorm,
                    pdf_ratio_normal_truncated)

try:
    from tqdm import trange
except(ImportError):
    print("tqdm module not found, install it to see a progress bar during preprocessing.")
    trange = range

from utils import get_samples

here = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, os.path.join(here, ".."))

from prior import (VarConfig,
                   UniformPrior,
                   RatioUniformPrior,
                   comp_variables_dict)

sys.path.insert(0, os.path.join(here, "..", "surrogate"))
sys.path.insert(0, os.path.join(here, "..", "surrogate", 'nn_surrogate')) # beacuse of pickle

from nn_surrogate import Surrogate

RA = np.sqrt(135/(4*np.pi))

def run_experiment_equilibrium(k,
                               *,
                               surrogate: Surrogate,
                               phase_1_file: str,
                               phase_2_file: str,
                               nsamples: int,
                               res_name: str,
                               D_ref:    float=7.82,
                               hmin_ref: float=0.81,
                               hmax_ref: float=2.58):

    conf_v   = comp_variables_dict['v']
    conf_mu  = comp_variables_dict['mu']
    conf_kb  = comp_variables_dict['kb']
    conf_FvK = comp_variables_dict['FvK']

    psi_samples = get_samples(phase_2_file)
    theta_samples = get_samples(phase_1_file)

    # samples of the posterior of hyperparameters (from phase 2)
    mu_v   = psi_samples["mu_v"]
    sig_v  = psi_samples["sig_v"]
    mu_mu  = psi_samples["mu_mu"]
    sig_mu = psi_samples["sig_mu"]
    mu_kb  = psi_samples["mu_kb"]
    sig_kb = psi_samples["sig_kb"]

    # samples from the posterior parameters of single level (from phase 1)
    v_1   = theta_samples['v']
    FvK_1 = theta_samples['FvK']

    # precompute bik
    b = np.zeros_like(mu_v)

    p_v_1 = conf_v.prior.pdf(v_1)
    p_FvK_1 = conf_FvK.prior.pdf(FvK_1)

    print("Preprocessing for equilibrium experiment:")
    for i in trange(len(b)):
        p_v_1_phi = pdf_truncnorm(v_1, a=0, b=1, loc=mu_v[i], scale=sig_v[i])
        p_FvK_1_phi = pdf_ratio_normal_truncated(FvK_1,
                                                 mux=mu_mu[i], sigx=sig_mu[i],
                                                 ay=conf_kb.low()/RA**2, by=conf_kb.high()/RA**2,
                                                 muy=mu_kb[i]/RA**2, sigy=sig_kb[i]/RA**2)

        b[i] = np.mean(p_v_1_phi * p_FvK_1_phi / (p_v_1 * p_FvK_1))
        b[i] = max([1e-6, b[i]]) # TODO



    def log_likelihood(ks):
        v, FvK, sigma = ks['Parameters']

        mu = 4 # arbitrary (non sensitive)
        b2 = 2 # arbitrary (non sensitive)

        D, hmin, hmax = surrogate.evaluate_equilibrium([v, mu, FvK, b2])

        log_p_data_knowing_theta = norm.logpdf(D_ref, loc=D, scale=D*sigma) \
            + norm.logpdf(hmin_ref, loc=hmin, scale=hmin*sigma) \
            + norm.logpdf(hmax_ref, loc=hmax, scale=hmax*sigma)

        p_v_phi = pdf_truncnorm(v, a=0, b=1, loc=mu_v, scale=sig_v)
        p_FvK_phi = pdf_ratio_normal_truncated(FvK,
                                               mux=mu_mu[i], sigx=sig_mu[i],
                                               ay=conf_kb.low()/RA**2, by=conf_kb.high()/RA**2,
                                               muy=mu_kb[i]/RA**2, sigy=sig_kb[i]/RA**2)

        l = log_p_data_knowing_theta + np.log(np.mean(p_v_phi * p_FvK_phi / b))

        ks["logLikelihood"] = l


    e = korali.Experiment()

    e["Problem"]["Type"] = "Bayesian/Custom"
    e["Problem"]["Likelihood Model"] = log_likelihood

    e["Solver"]["Type"] = "Sampler/TMCMC"
    e["Solver"]["Population Size"] = nsamples
    e["Solver"]["Target Coefficient Of Variation"] = 0.6

    variables = [comp_variables_dict['v'],
                 comp_variables_dict['FvK'],
                 VarConfig(name="sigma",
                           prior=UniformPrior(a=1e-4, b=0.5))]

    for i, var in enumerate(variables):
        var.configure_korali(e, i)

    e["File Output"]["Path"] = res_name
    e["Console Output"]["Verbosity"] = "Detailed"

    k.run(e)



def run_experiment_stretching(k, *,
                              surrogate: Surrogate,
                              phase_1_file: str,
                              phase_2_file: str,
                              nsamples: int,
                              res_name: str,
                              csv_ref: str):

    df = pd.read_csv(csv_ref)

    Fext   = df["Fext"].tolist()
    D0_ref = df["D0"].to_numpy()
    D1_ref = df["D1"].to_numpy()

    conf_v  = comp_variables_dict['v']
    conf_mu = comp_variables_dict['mu']
    conf_kb = comp_variables_dict['kb']
    conf_b2 = comp_variables_dict['b2']

    psi_samples   = get_samples(phase_2_file)
    theta_samples = get_samples(phase_1_file)

    # samples of the posterior of hyperparameters (from phase 2)
    mu_v   = psi_samples["mu_v"]
    sig_v  = psi_samples["sig_v"]
    mu_mu  = psi_samples["mu_mu"]
    sig_mu = psi_samples["sig_mu"]
    mu_kb  = psi_samples["mu_kb"]
    sig_kb = psi_samples["sig_kb"]
    mu_b2  = psi_samples["mu_b2"]
    sig_b2 = psi_samples["sig_b2"]

    # samples from the posterior parameters of single level (from phase 1)
    v_1  = theta_samples['v']
    mu_1 = theta_samples['mu']
    kb_1 = theta_samples['kb']
    b2_1 = theta_samples['b2']

    # precompute bik
    b = np.zeros_like(mu_v)

    p_v_1  = conf_v.prior.pdf(v_1)
    p_mu_1 = conf_mu.prior.pdf(mu_1)
    p_kb_1 = conf_kb.prior.pdf(kb_1)
    p_b2_1 = conf_b2.prior.pdf(b2_1)

    print(f"Preprocessing for stretching experiment ({os.path.basename(csv_ref)}):")
    for i in trange(len(b)):
        p_v_1_phi  = pdf_truncnorm(v_1, a=0, b=1, loc=mu_v[i], scale=sig_v[i])
        p_mu_1_phi = pdf_norm(mu_1, loc=mu_mu[i], scale=sig_mu[i])
        p_kb_1_phi = pdf_truncnorm(kb_1, a=conf_kb.low(), b=conf_kb.high(), loc=mu_kb[i], scale=sig_kb[i])
        p_b2_1_phi = pdf_norm(b2_1, loc=mu_b2[i], scale=sig_b2[i])
        b[i] = np.mean(p_v_1_phi * p_mu_1_phi * p_kb_1_phi * p_b2_1_phi / (p_v_1 * p_mu_1 * p_kb_1 * p_b2_1))
        b[i] = max([1e-6, b[i]]) # TODO

    def log_likelihood(ks):
        v, mu, kb, b2, sigma0, sigma1 = ks['Parameters']
        FvK = mu * RA**2 / kb

        D0, D1 = surrogate.evaluate_stretching([v, mu, FvK, b2], Fext)
        D0 = np.array(D0)
        D1 = np.array(D1)

        log_p_data_knowing_theta = \
            + np.sum(norm.logpdf(D0_ref, loc=D0, scale=sigma0)) \
            + np.sum(norm.logpdf(D1_ref, loc=D1, scale=sigma1))

        p_v_phi  = pdf_truncnorm(v, a=0, b=1, loc=mu_v, scale=sig_v)
        p_mu_phi = pdf_norm(mu, loc=mu_mu, scale=sig_mu)
        p_kb_phi = pdf_truncnorm(kb, a=conf_kb.low(), b=conf_kb.high(), loc=mu_kb, scale=sig_kb)
        p_b2_phi = pdf_norm(b2, loc=mu_b2, scale=sig_b2)

        l = log_p_data_knowing_theta + np.log(np.mean((p_v_phi * p_mu_phi * p_kb_phi * p_b2_phi) / b))

        ks["logLikelihood"] = l


    e = korali.Experiment()

    e["Problem"]["Type"] = "Bayesian/Custom"
    e["Problem"]["Likelihood Model"] = log_likelihood

    e["Solver"]["Type"] = "Sampler/TMCMC"
    e["Solver"]["Population Size"] = nsamples
    e["Solver"]["Target Coefficient Of Variation"] = 0.8
    e["Solver"]["Covariance Scaling"] = 0.04

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

    k.run(e)





def run_experiment_relaxation(k, *,
                              surrogate: Surrogate,
                              phase_1_file: str,
                              phase_2_file: str,
                              nsamples: int,
                              res_name: str,
                              csv_ref: str):


    df = pd.read_csv(csv_ref)

    t = df["t"].to_numpy()
    y = df["L_W"].to_numpy()

    conf_mu   = comp_variables_dict['mu']
    conf_etam = comp_variables_dict['etam']
    conf_etam_mu = VarConfig(name="etam_mu",
                             prior=RatioUniformPrior(a=conf_etam.low(), b=conf_etam.high(),
                                                     c=conf_mu.low(), d=conf_mu.high()))
    psi_samples = get_samples(phase_2_file)
    theta_samples = get_samples(phase_1_file)

    # samples of the posterior of hyperparameters (from phase 2)
    mu_etam_mu  = psi_samples["mu_etam_mu"]
    sig_etam_mu = psi_samples["sig_etam_mu"]

    # samples from the posterior parameters of single level (from phase 1)
    etam_mu_1 = theta_samples['etam_mu']

    # precompute bik
    b = np.zeros_like(mu_etam_mu)

    p_etam_mu_1 = conf_etam_mu.prior.pdf(etam_mu_1)

    print(f"Preprocessing for relaxation experiment ({os.path.basename(csv_ref)}):")
    for i in trange(len(b)):
        p_etam_mu_1_phi  = norm.pdf(etam_mu_1, loc=mu_etam_mu[i], scale=sig_etam_mu[i])
        b[i] = np.mean(p_etam_mu_1_phi / (p_etam_mu_1))
        b[i] = max([1e-6, b[i]]) # TODO

    def log_likelihood(ks):
        etam_mu, sigma_tc, tc, z0, zinf, sigma_z = ks["Parameters"]
        v = 0.95
        mu = 4
        FvK = 250
        b2 = 1.6
        etam = etam_mu * mu

        # log likelihood of data knowing theta
        z = exponential_model(z0, zinf, tc, t)
        log_likelihood = np.sum(norm.logpdf(y, loc=z, scale=sigma_z))
        tc_sim = surrogate.evaluate_relax([v, mu, FvK, b2, etam])
        log_likelihood += norm.logpdf(tc, loc=tc_sim, scale=tc*sigma_tc)

        # theta knowing psi contribution
        p_etam_mu_phi = norm.pdf(etam_mu, loc=mu_etam_mu, scale=sig_etam_mu)
        log_likelihood += np.log(np.mean(p_etam_mu_phi / b))

        ks["logLikelihood"] = log_likelihood


    e = korali.Experiment()

    e["Problem"]["Type"] = "Bayesian/Custom"
    e["Problem"]["Likelihood Model"] = log_likelihood

    e["Solver"]["Type"] = "Sampler/TMCMC"
    e["Solver"]["Population Size"] = nsamples
    e["Solver"]["Target Coefficient Of Variation"] = 0.6
    e["Solver"]["Covariance Scaling"] = 0.04

    variables = [conf_etam_mu,
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



if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('surr_dir', type=str, help="The directory that contains the trained surrogate state.")
    parser.add_argument('results_phase_1_dir', type=str, help="The results of the single-level parameters inference (phase 1)")
    parser.add_argument('results_phase_2_dir', type=str, help="The results of the hyper-parameters inference (phase 2)")
    parser.add_argument('--nsamples', type=int, default=10000, help="Number of samples.")
    parser.add_argument('--stretch-exp', type=str, nargs='+', default=[], help="Stretching experiment files.")
    parser.add_argument('--relax-exp', type=str, nargs='+', default=[], help="Stretching experiment files.")
    parser.add_argument('--output-dir', type=str, default="results_phase_3_exp", help="The directory that will contain the results.")
    parser.add_argument('--cores', type=int, default=4, help="Number of cores to use.")
    args = parser.parse_args()

    surrogate = Surrogate(args.surr_dir)
    phase_2_file = os.path.join(args.results_phase_2_dir, 'latest')
    nsamples = args.nsamples

    k = korali.Engine()
    k["Conduit"]["Type"] = "Concurrent"
    k["Conduit"]["Concurrent Jobs"] = args.cores

    run_experiment_equilibrium(k,
                               surrogate=surrogate,
                               phase_1_file=os.path.join(args.results_phase_1_dir, 'eq', 'latest'),
                               phase_2_file=phase_2_file,
                               nsamples=nsamples,
                               res_name=os.path.join(args.output_dir, 'eq'))

    for i, f in enumerate(args.stretch_exp):
        run_experiment_stretching(k,
                                  surrogate=surrogate,
                                  phase_1_file=os.path.join(args.results_phase_1_dir, f'stretch_{i}', 'latest'),
                                  phase_2_file=phase_2_file,
                                  nsamples=nsamples,
                                  res_name=os.path.join(args.output_dir, f'stretch_{i}'),
                                  csv_ref=f)

    for i, f in enumerate(args.relax_exp):
        run_experiment_relaxation(k,
                                  surrogate=surrogate,
                                  phase_1_file=os.path.join(args.results_phase_1_dir, f'relax_{i}', 'latest'),
                                  phase_2_file=phase_2_file,
                                  nsamples=args.nsamples,
                                  res_name=os.path.join(args.output_dir, f"relax_{i}"),
                                  csv_ref=f)
