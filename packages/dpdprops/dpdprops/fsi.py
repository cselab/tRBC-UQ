import math

from .dpdparams import DPDParams

def get_gamma_fsi_DPD_membrane(*,
                               eta: float,
                               kpow: float,
                               nd_membrane: float,
                               nd_fluid: float,
                               rc: float) -> float:
    """
    Compute DPD dissipation coefficient for interactions between DPD solvent and membrane.
    Reference: Fedosov2010, doctoral thesis.

    Arguments:
        eta: Dynamic viscosity of the solvent interacting with the membrane.
        kpow: DPD kernel exponent.
        nd_membrane: The 2D number density of the membrane (1/L^2 units)
        nd_fluid:  The number density of solvent (1/L^3 units)
        rc: The cutoff radius of the DPD interaction.

    Returns:
        The dissipation coefficient for the DPD interaction between the
        membrane particles and the solvent particles.
    """
    s = 2 * kpow
    I = 3 * math.pi * rc**4 / (2 * (s+1) * (s+2) * (s+3) * (s+4))
    g_fsi = eta / (nd_fluid * nd_membrane * I)
    return g_fsi


def create_fsi_dpd_params(*,
                          fluid_params: 'DPDParams',
                          nd_membrane: float) -> 'DPDParams':
    """
    Create DPD parameters for FSI between a fluid and a membrane made of particles.
    The conservative coefficient is set to zero and the dissipative coefficient is
    that returned by get_gamma_fsi_DPD_membrane.

    Arguments:
        fluid_params: The DPDParams of the solvent.
        nd_membrane: The 2D number density of the particles forming the membrane.

    Returns:
        The DPDParams of the FSI.
    """

    gamma = get_gamma_fsi_DPD_membrane(eta=fluid_params.dynamic_viscosity(),
                                       nd_fluid=fluid_params.nd,
                                       kpow=fluid_params.kpow,
                                       rc=fluid_params.rc,
                                       nd_membrane=nd_membrane)

    return DPDParams(a=fluid_params.a * 0, # in case we use pint, it will keep the units.
                     gamma=gamma,
                     kBT=fluid_params.kBT,
                     nd=fluid_params.nd,
                     mass=fluid_params.mass,
                     rc=fluid_params.rc,
                     kpow=fluid_params.kpow)
