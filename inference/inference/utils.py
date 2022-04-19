#!/usr/bin env python

import json
import numpy as np

def get_samples(korali_file,
                extract_likelihood: bool=True):
    """
    Read the samples from the json file stored by korali.
    The data is returned in a dictionary of the form {'variable name': np.array}
    Arguments:
        korali_file: the json file produced by korali.
        extract_likelihood: if True, extract the log likelihood and store it the returned data.
    """

    with open(korali_file) as f:
        data = json.load(f)

    varnames = [var['Name'] for var in data["Variables"]]
    params = np.array(data["Results"]["Sample Database"])

    samples = {}
    for i, varname in enumerate(varnames):
        samples[varname] = params[:,i]

    if extract_likelihood:
        log_likelihood = np.array(data["Solver"]["Sample LogLikelihood Database"])
        samples["log_likelihood"] = log_likelihood

    return samples
