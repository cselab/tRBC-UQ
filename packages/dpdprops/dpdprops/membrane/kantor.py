#!/usr/bin/env python

def bending_modulus_to_kantor(*, kc: float) -> float:
    """
    Compute the bending coefficent in the Kantor model.
    See Fedosov et al., Biophys. J., 2010.

    Parameters:
        kc: Macroscopic bending modulus.
    Returns:
        The bending coefficient in the Kantor model.
    """
    kb_Kantor = 2 * kc / 3**(1/2)
    return kb_Kantor


def kantor_to_bending_modulus(*, kb):
    """
    Compute the bending modulus from the Kantor bending coeff.
    See Fedosov et al., Biophys. J., 2010.

    Parameters:
        kb: The bending coefficient in the Kantor model.
    Returns:
        Macroscopic bending modulus.
    """
    return kb * 3**(1/2) / 2
