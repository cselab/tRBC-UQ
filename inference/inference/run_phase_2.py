#! /usr/bin/env python

import glob
import korali
import numpy as np
import os
import sys

from common import (pdf_norm,
                    pdf_truncnorm,
                    pdf_ratio_normal_truncated)
from utils import get_samples

here = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, os.path.join(here, ".."))

from prior import (comp_variables_dict,
                   VarConfig,
                   RatioUniformPrior)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('results_phase_1_dir', type=str, help="The results of the single level inference (phase 1)")
    parser.add_argument('--nsamples', type=int, default=10000, help="Number of samples.")
    parser.add_argument('--output-dir', type=str, default="results_phase_2", help="The directory that will contain the results.")
    parser.add_argument('--cores', type=int, default=4, help="Number of cores to use.")
    args = parser.parse_args()

    base_dir = args.results_phase_1_dir

    ndatasets_st = len(glob.glob(os.path.join(base_dir, "stretch*")))
    ndatasets_re = len(glob.glob(os.path.join(base_dir, "relax*")))

    samples_eq = []
    samples_st = []
    samples_re = []

    samples_eq.append(get_samples(os.path.join(base_dir, "eq", 'latest')))

    for i in range(ndatasets_st):
        samples_st.append(get_samples(os.path.join(base_dir, f"stretch_{i}", 'latest')))

    for i in range(ndatasets_re):
        samples_re.append(get_samples(os.path.join(base_dir, f"relax_{i}", 'latest')))

    conf_v   = comp_variables_dict['v']
    conf_mu  = comp_variables_dict['mu']
    conf_kb  = comp_variables_dict['kb']
    conf_FvK = comp_variables_dict['FvK']
    conf_b2  = comp_variables_dict['b2']
    conf_etam = comp_variables_dict['etam']
    conf_etam_mu = VarConfig(name="etam_mu",
                             prior=RatioUniformPrior(a=conf_etam.low(), b=conf_etam.high(),
                                                     c=conf_mu.low(), d=conf_mu.high()))

    RA = np.sqrt(135 / (4*np.pi))

    def log_likelihood(ks):
        mu_v, sig_v, mu_mu, sig_mu, mu_kb, sig_kb, mu_b2, sig_b2, mu_etam_mu, sig_etam_mu = ks["Parameters"]
        l = 0

        for s in samples_eq:
            v   = s['v']
            FvK = s['FvK']

            p_v_phi   = pdf_truncnorm(v, a=0, b=1, loc=mu_v, scale=sig_v)
            p_FvK_phi = pdf_ratio_normal_truncated(FvK,
                                                   mux=mu_mu, sigx=sig_mu,
                                                   ay=conf_kb.low()/RA**2,
                                                   by=conf_kb.high()/RA**2,
                                                   muy=mu_kb/RA**2, sigy=sig_kb/RA**2)

            p_v   = conf_v.prior.pdf(v)
            p_FvK = conf_FvK.prior.pdf(FvK)

            l += np.log(np.mean(p_v_phi * p_FvK_phi / (p_v * p_FvK)))

        for s in samples_st:
            v  = s['v']
            mu = s['mu']
            kb = s['kb']
            b2 = s['b2']

            p_v_phi  = pdf_truncnorm(v,  a=0,  b=1,  loc=mu_v,  scale=sig_v)
            p_mu_phi = pdf_norm(mu, loc=mu_mu, scale=sig_mu)
            p_kb_phi = pdf_truncnorm(kb, a=conf_kb.low(), b=conf_kb.high(), loc=mu_kb, scale=sig_kb)
            p_b2_phi = pdf_norm(b2, loc=mu_b2, scale=sig_b2)

            p_v  = conf_v.prior.pdf(v)
            p_mu = conf_mu.prior.pdf(mu)
            p_kb = conf_kb.prior.pdf(kb)
            p_b2 = conf_b2.prior.pdf(b2)

            l += np.log(np.mean(p_v_phi * p_mu_phi * p_kb_phi * p_b2_phi / (p_v * p_mu * p_kb * p_b2)))

        for s in samples_re:
            etam_mu = s['etam_mu']
            p_etam_mu_phi = pdf_norm(etam_mu, loc=mu_etam_mu, scale=sig_etam_mu)
            p_etam_mu = conf_etam_mu.prior.pdf(etam_mu)

            l += np.log(np.mean(p_etam_mu_phi / (p_etam_mu)))

        ks["logLikelihood"] = l

    e = korali.Experiment()
    e["Problem"]["Type"] = "Bayesian/Custom"
    e["Problem"]["Likelihood Model"] = log_likelihood

    e["Variables"] = [
        {"Name": "mu_v",
         "Prior Distribution": "Prior mu_v"},
        {"Name": "sig_v",
         "Prior Distribution": "Prior sig_v"},

        {"Name": "mu_mu",
         "Prior Distribution": "Prior mu_mu"},
        {"Name": "sig_mu",
         "Prior Distribution": "Prior sig_mu"},

        {"Name": "mu_kb",
         "Prior Distribution": "Prior mu_kb"},
        {"Name": "sig_kb",
         "Prior Distribution": "Prior sig_kb"},

        {"Name": "mu_b2",
         "Prior Distribution": "Prior mu_b2"},
        {"Name": "sig_b2",
         "Prior Distribution": "Prior sig_b2"},

        {"Name": "mu_etam_mu",
         "Prior Distribution": "Prior mu_etam_mu"},
        {"Name": "sig_etam_mu",
         "Prior Distribution": "Prior sig_etam_mu"}
    ]

    e["Distributions"] = [
        {"Name": "Prior mu_v",
         "Type": "Univariate/Uniform",
         "Minimum": conf_v.low(),
         "Maximum": conf_v.high()},
        {"Name": "Prior sig_v",
         "Type": "Univariate/Uniform",
         "Minimum": 1e-2,
         "Maximum": 0.5},

        {"Name": "Prior mu_mu",
         "Type": "Univariate/Uniform",
         "Minimum": conf_mu.low(),
         "Maximum": conf_mu.high()},
        {"Name": "Prior sig_mu",
         "Type": "Univariate/Uniform",
         "Minimum": 1e-2,
         "Maximum": 10.0},

        {"Name": "Prior mu_kb",
         "Type": "Univariate/Uniform",
         "Minimum": conf_kb.low(),
         "Maximum": conf_kb.high()},
        {"Name": "Prior sig_kb",
         "Type": "Univariate/Uniform",
         "Minimum": 1e-2,
         "Maximum": 0.5},

        {"Name": "Prior mu_b2",
         "Type": "Univariate/Uniform",
         "Minimum": conf_b2.low(),
         "Maximum": conf_b2.high()},
        {"Name": "Prior sig_b2",
         "Type": "Univariate/Uniform",
         "Minimum": 1e-2,
         "Maximum": 10.0},

        {"Name": "Prior mu_etam_mu",
         "Type": "Univariate/Uniform",
         "Minimum": 0.01,
         "Maximum": 0.4},
        {"Name": "Prior sig_etam_mu",
         "Type": "Univariate/Uniform",
         "Minimum": 1e-2,
         "Maximum": 0.5},
    ]

    e["Solver"]["Type"] = "Sampler/TMCMC"
    e["Solver"]["Population Size"] = args.nsamples
    e["Solver"]["Target Coefficient Of Variation"] = 0.4
    e["Solver"]["Covariance Scaling"] = 0.06

    e["Console Output"]["Verbosity"] = "Detailed"
    e["File Output"]["Path"] = args.output_dir
    e["Random Seed"] = 0xC0FFEE

    k = korali.Engine()
    k["Conduit"]["Type"] = "Concurrent"
    k["Conduit"]["Concurrent Jobs"] = args.cores
    k.run(e)
