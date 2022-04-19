#!/usr/bin/env python

description = """
Extract the samples mean, standard deviation and quantiles from korali output files
"""

import numpy as np
import os
import pint
import sys

from utils import get_samples

here = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, os.path.join(here, ".."))

from prior import (VarConfig,
                   UniformPrior,
                   RatioUniformPrior,
                   comp_variables_dict)


def main(argv):
    import argparse
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('src', type=str, help="Input json file.")
    args = parser.parse_args(argv)

    ureg = pint.UnitRegistry()
    data = get_samples(args.src)

    A_ = 135 * ureg.um**2
    RA_ = np.sqrt(A_ / (4*np.pi)).to('um')

    confidences = [0.5, 0.75, 0.90, 0.99]

    for key, val in data.items():
        if key == "log_likelihood":
            continue

        print(key)
        print(f"mean {np.mean(val)}")
        print(f"median {np.median(val)}")
        print(f"std  {np.std(val)}")
        for c in confidences:
            q = (1 - c)/2
            print(f"{c*100}% confidence interval: [{np.quantile(val, q)}, {np.quantile(val, 1-q)}]")


        print()


    if 'mu' in data and 'FvK' in data:
        mu_ = data['mu'] * ureg.uN / ureg.m
        FvK = data['FvK']
        kb_ = (mu_ * RA_**2 / FvK).to(ureg.J)

        print('kb')
        print(f"mean {np.mean(kb_)}")
        print(f"median {np.median(kb_)}")
        print(f"std  {np.std(kb_)}")
        for c in confidences:
            q = (1 - c)/2
            print(f"{c*100}% confidence interval: [{np.quantile(kb_, q)}, {np.quantile(kb_, 1-q)}]")
        print()

    if 'mu' in data and 'etam_mu' in data:
        mu_ = data['mu'] * ureg.uN / ureg.m
        etam_mu_ = data['etam_mu'] * ureg.s
        etam_ = (mu_ * etam_mu_).to(ureg.Pa * ureg.s * ureg.um)

        print('etam')
        print(f"mean {np.mean(etam_)}")
        print(f"median {np.median(etam_)}")
        print(f"std  {np.std(etam_)}")
        for c in confidences:
            q = (1 - c)/2
            print(f"{c*100}% confidence interval: [{np.quantile(etam_, q)}, {np.quantile(etam_, 1-q)}]")
        print()



    if "log_likelihood" in data.keys():
        log_likelihood = data["log_likelihood"]
        ibest = np.argmax(log_likelihood)

        print("Max Likelihood")
        for key in data.keys():
            print(f"{key}: {data[key][ibest]}")
        if 'mu' in data and 'FvK' in data.keys():
            mu_ = data['mu'][ibest] * ureg.uN / ureg.m
            FvK = data['FvK'][ibest]
            kb_ = ((mu_ * RA_**2) / FvK).to(ureg.J)
            print(f"kb: {kb_}")
        if 'mu' in data and 'etam_mu' in data.keys():
            mu_ = data['mu'][ibest] * ureg.uN / ureg.m
            etam_mu_ = data['etam_mu'][ibest] * ureg.s
            etam_ = (mu_ * etam_mu_).to(ureg.Pa * ureg.s * ureg.um)
            print(f"etam: {etam_}")
        print()


        if 'thetanew' in args.src:
            conf_variables = {
                "v": comp_variables_dict["v"],
                "mu": VarConfig(name="mu",
                                prior=UniformPrior(a=0, b=20)),
                "kb": comp_variables_dict["kb"],
                "b2": VarConfig(name="b2",
                                prior=UniformPrior(a=0, b=10)),
                "etam_mu": VarConfig(name="etam_mu",
                                     prior=UniformPrior(a=0, b=2))
            }

        else:
            conf_variables = comp_variables_dict

            conf_mu = comp_variables_dict['mu']
            conf_etam = comp_variables_dict['etam']
            conf_variables["etam_mu"] = VarConfig(name="etam_mu",
                                                  prior=RatioUniformPrior(a=conf_etam.low(),
                                                                          b=conf_etam.high(),
                                                                          c=conf_mu.low(),
                                                                          d=conf_mu.high()))

        print("Maximum A Posteriori")
        log_MAP = log_likelihood
        for key, values in data.items():
            if key in conf_variables:
                conf_var = conf_variables[key]
                log_MAP += conf_var.prior.logpdf(values)
            elif key != "log_likelihood":
                print(f"skipping {key}")

        imap = np.argmax(log_MAP)
        for key in data.keys():
            print(f"{key}: {data[key][imap]}")
        if 'mu' in data and 'FvK' in data.keys():
            mu_ = data['mu'][imap] * ureg.uN / ureg.m
            FvK = data['FvK'][imap]
            kb_ = ((mu_ * RA_**2) / FvK).to(ureg.J)
            print(f"kb: {kb_}")
        if 'mu' in data and 'etam_mu' in data.keys():
            mu_ = data['mu'][imap] * ureg.uN / ureg.m
            etam_mu_ = data['etam_mu'][imap] * ureg.s
            etam_ = (mu_ * etam_mu_).to(ureg.Pa * ureg.s * ureg.um)
            print(f"etam: {etam_}")
        print()





if __name__ == '__main__':
    main(sys.argv[1:])
