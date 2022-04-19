#!/usr/bin/env python

import math
from .selfdiffusion import compute_self_diffusion

def compute_kinematic_viscosity(*,
                                kpow: float,
                                nd: float,
                                mass: float,
                                kBT: float,
                                gamma: float,
                                rc: float):
    """
    Compute the kinematic viscosity of a DPD fluid.
    This approximation holds only for a RDF g(r) ~ 1.
    See Groot & Warren, 1997 (appendix).

    Arguments:
        kpow: Exponent used in the random force kernel.
        nd: Number density.
        mass: Mass of one DPD particle.
        kBT: Temperature.
        gamma: Dissipative coefficient.
        rc: Cutoff radius.
    Return:
        The kinematic viscosity of the fluid.
    """
    D = compute_self_diffusion(kpow=kpow, nd=nd, kBT=kBT, gamma=gamma, rc=rc)
    nuK = D / 2

    s = kpow * 2
    nuD = 16 * math.pi * gamma / mass * nd * rc**5 / (5 * (s+1) * (s+2) * (s+3) * (s+4) * (s+5))

    return nuK + nuD


def compute_dynamic_viscosity(*,
                              kpow: float,
                              nd: float,
                              mass: float,
                              kBT: float,
                              gamma: float,
                              rc: float):
    """
    Compute the dynamic viscosity of a DPD fluid.
    This approximation holds only for a RDF g(r) ~ 1.
    See Groot & Warren, 1997 (appendix).

    Arguments:
        kpow: Exponent used in the random force kernel.
        nd: Number density.
        mass: Mass of one DPD particle.
        kBT: Temperature.
        gamma: Dissipative coefficient.
        rc: Cutoff radius.
    Return:
        The kinematic viscosity of the fluid.
    """
    nu = compute_kinematic_viscosity(kpow=kpow, nd=nd, mass=mass, kBT=kBT, gamma=gamma, rc=rc)
    return nd * mass * nu
