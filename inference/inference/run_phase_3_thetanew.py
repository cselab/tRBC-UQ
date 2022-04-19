#!/usr/bin/env python

import korali
import numpy as np
import os
import sys

from common import (pdf_norm,
                    pdf_truncnorm)
from utils import get_samples

here = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, os.path.join(here, ".."))

from prior import (comp_variables_dict,
                   VarConfig,
                   UniformPrior,
                   RatioUniformPrior)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('results_phase_2_dir', type=str, help="The results of the hyper-parameters inference (phase 2)")
    parser.add_argument('--nsamples', type=int, default=10000, help="Number of samples.")
    parser.add_argument('--output-dir', type=str, default="results_phase_3_thetanew", help="The directory that will contain the results.")
    parser.add_argument('--cores', type=int, default=4, help="Number of cores to use.")
    args = parser.parse_args()

    samples = get_samples(os.path.join(args.results_phase_2_dir, 'latest'))

    conf_v  = comp_variables_dict["v"]
    conf_mu = comp_variables_dict["mu"]
    conf_kb = comp_variables_dict["kb"]
    conf_b2 = comp_variables_dict["b2"]

    def log_likelihood(ks):
        v, mu, kb, b2, etam_mu = ks["Parameters"]

        mu_v   = samples["mu_v"]
        sig_v  = samples["sig_v"]
        mu_mu  = samples["mu_mu"]
        sig_mu = samples["sig_mu"]
        mu_kb  = samples["mu_kb"]
        sig_kb = samples["sig_kb"]
        mu_b2  = samples["mu_b2"]
        sig_b2 = samples["sig_b2"]
        mu_etam_mu  = samples["mu_etam_mu"]
        sig_etam_mu = samples["sig_etam_mu"]

        p_v_phi       = pdf_truncnorm(v, a=0, b=1, loc=mu_v, scale=sig_v)
        p_mu_phi      = pdf_norm(mu, loc=mu_mu, scale=sig_mu)
        p_kb_phi      = pdf_truncnorm(kb, a=conf_kb.low(), b=conf_kb.high(), loc=mu_kb, scale=sig_kb)
        p_b2_phi      = pdf_norm(b2, loc=mu_b2, scale=sig_b2)
        p_etam_mu_phi = pdf_norm(etam_mu, loc=mu_etam_mu, scale=sig_etam_mu)

        ks["logLikelihood"] = np.log(np.mean(p_v_phi * p_mu_phi * p_kb_phi * p_b2_phi * p_etam_mu_phi))


    e = korali.Experiment()
    e["Problem"]["Type"] = "Bayesian/Custom"
    e["Problem"]["Likelihood Model"] = log_likelihood

    conf_mu   = comp_variables_dict["mu"]
    conf_etam = comp_variables_dict["etam"]

    # variables = [
    #     comp_variables_dict["v"],
    #     comp_variables_dict["mu"],
    #     comp_variables_dict["kb"],
    #     comp_variables_dict["b2"],
    #     VarConfig(name="etam_mu",
    #               prior=RatioUniformPrior(a=conf_etam.low(), b=conf_etam.high(),
    #                                       c=conf_mu.low(), d=conf_mu.high()))
    # ]
    # Here we need "large enough" bounds, so we do it manually.
    variables = [
        comp_variables_dict["v"],
        VarConfig(name="mu",
                  prior=UniformPrior(a=0, b=20)),
        comp_variables_dict["kb"],
        VarConfig(name="b2",
                  prior=UniformPrior(a=0, b=10)),
        VarConfig(name="etam_mu",
                  prior=UniformPrior(a=0, b=2))
    ]

    for i, var in enumerate(variables):
        var.configure_korali(e, i)

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
