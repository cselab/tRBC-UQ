#!/usr/bin/env python

def shear_modulus_to_wlc(*,
                         mu: float,
                         l0: float,
                         x0: float,
                         m: float=2) -> float:
    """
    Compute the force coefficient ks in the wlc model for shear forces.
    See Fedosov et al., Biophys. J., 2010.

    Parameters:
        mu: Macroscopic shear modulus.
        l0: Average equilibrium length of the spring.
        x0: l0/lmax, a computational parameter of the model to calibrate.
        m: Exponent of the chain, a computational parameter of the model to calibrate.

    Returns:
        The spring coefficient ks used in the wlc model.
    """
    lm = l0/x0
    A = (6*pow(l0,(m+1))*pow(lm,2)-9*pow(l0,(m+2))*lm+4*pow(l0,(m+3))) / (4*pow(lm,3)-8*l0*pow(lm,2)+4*pow(l0,2)*lm)
    B = 3**0.5 * (m+1)
    C = (4.*pow(l0,(m+1)))
    CC = (x0/(2.*pow((1-x0),3)) - 1./(4.*pow((1-x0),2)) + 1./4) * 3**0.5 / (4.*l0)
    ks = mu / ( CC + A*B/C )
    return ks


def wlc_to_shear_modulus(*,
                         ks: float,
                         l0: float,
                         x0: float,
                         m: float=2) -> float:
    """
    Compute the shear modulus mu from the wlc model parameters.
    See Fedosov et al., Biophys. J., 2010.

    Parameters:
        ks: The spring coefficient ks used in the wlc model.
        l0: Average equilibrium length of the spring.
        x0: l0/lmax, a computational parameter of the model to calibrate.
        m: Exponent of the chain, a computational parameter of the model to calibrate.

    Returns:
        Macroscopic shear modulus.
    """
    lm = l0 / x0
    kp = ks * l0**(m+1) * (6 * lm**2 - 9 * lm * l0 + 4 * l0**2) / (4 * lm**3 - 8*l0*lm**2 + 4 * l0**2 * lm)
    mu = 3**(1/2) * ks / (4 * l0) * (x0 / (2 * (1-x0)**3) - 1 / (4 * (1-x0)**2) + 1/4) \
        + 3 * 3**(1/2) * kp / (4 * l0**3)
    return mu
