#!/usr/bin/env python

import math

def compute_self_diffusion(*,
                           kpow: float,
                           nd: float,
                           kBT: float,
                           gamma: float,
                           rc: float):
    """
    Compute the self diffusion coefficient of a DPD fluid.
    This approximation holds only for a RDF g(r) ~ 1.
    See Groot & Warren, 1997 (appendix).

    Arguments:
        kpow: Exponent used in the random force kernel.
        nd: Number density.
        kBT: Temperature.
        gamma: Dissipative coefficient.
        rc: Cutoff radius.
    Return:
        The self diffusion of the fluid.
    """
    s = kpow * 2
    return (s+1) * (s+2) * (s+3) * 3 * kBT / (8 * math.pi * nd * gamma * rc**3)
