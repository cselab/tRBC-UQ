#!/usr/bin/env python

def membrane_gamma_to_membrane_viscosity(*,
                                        gC: float) -> float:
    """
    Compute the membrane viscosity from the force coefficient gammaC.
    See Fedosov et al., Biophys. J., 2010.

    Parameters:
        gC: the gamma parameter of the membrane.

    Returns:
        The 2D membrane viscosity.
    """
    return gC * 3**0.5 / 4

def membrane_viscosity_to_membrane_gamma(*,
                                         etam: float) -> float:
    """
    Compute the force coefficient gammaC from the 2D membrane viscosity.
    See Fedosov et al., Biophys. J., 2010.

    Parameters:
        etam: The 2D membrane viscosity.

    Returns:
        The gamma parameter of the membrane.
    """
    return etam * 4 / 3**0.5
