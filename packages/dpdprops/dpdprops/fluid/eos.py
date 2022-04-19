#!/usr/bin/env python

# fitted to the simulation.
# mean field value is pi/30 ~ 0.1047
dpd_eos_coeff = 0.103

# the eos is valid only for higher number densities than this lower bound.
nd_rc3_limit = 8

def compute_pressure(*,
                     a: float,
                     nd: float,
                     kBT: float,
                     rc: float):
    """
    Compute the pressure of the DPD fluid with the above properties.
    See Groot & Warren, 1997.
    The alpha coefficient is fitted from simulations.

    Arguments:
        a: conservative param
        nd: number density
        kBT: temperature
        rc: the cut-off radius
    Return:
        The pressure of the fluid.
    """

    if nd * rc**3 < nd_rc3_limit:
        raise ValueError(f"compute_pressure is not accurate for nd * rc^3 < {nd_rc3_limit}, got {nd*rc**3}.")
    return nd * kBT + dpd_eos_coeff * rc**4 * a * nd**2


def sound_speed(*,
                a: float,
                nd: float,
                kBT: float,
                rc: float,
                mass: float):
    """
    Compute the speed of sound of the DPD fluid with the above properties.
    See Groot & Warren, 1997.

    Arguments:
        a: conservative param
        nd: number density
        kBT: temperature
        rc: the cut-off radius
        mass: mass of a single particle
    Return:
        The speed of sound of the fluid.
    """
    if nd * rc**3 < nd_rc3_limit:
        raise ValueError(f"sound_speed is not accurate for nd * rc^3 < {nd_rc3_limit}, got {nd*rc**3}.")
    if a < 0:
        raise ValueError(f"repulsive coeff must be positive, got {a}.")

    return (kBT/mass + 2 * dpd_eos_coeff * rc**4 * a * nd / mass)**0.5
