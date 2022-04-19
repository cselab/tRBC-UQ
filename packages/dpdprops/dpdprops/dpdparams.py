#!/usr/bin/env python

from dataclasses import dataclass, asdict
import math
import warnings

from .fluid import *

@dataclass
class DPDParams:
    """
    DPD parameters class.
    Holds all parameters of a DPD fluid and translate them into fluid parameters
    such as mass density, speed of sounf, self diffusion and viscosity.
    """
    a: float
    gamma: float
    kBT: float
    nd: float
    mass: float
    rc: float
    kpow: float

    def sigma(self) -> float:
        """
        Compute sigma from the fluctuation disspation theorem.
        """
        return math.sqrt(2 * self.gamma * self.kBT)

    def to_interactions(self) -> dict:
        """
        get a dictionary that has the parameters as in Mirheo
        https://mirheo.readthedocs.io/en/latest/user/interactions.html#_mirheo.Interactions.Pairwise.__init__
        """
        return {"a": self.a,
                "gamma": self.gamma,
                "kBT": self.kBT,
                "power": self.kpow}

    def to_str(self) -> str:
        s = str()
        for key, val in asdict(self).items():
            if len(s):
                s += "_"
            s += f"{key}_{val}"
        return s

    def mass_density(self) -> float:
        return self.nd * self.mass

    def sound_speed(self) -> float:
        return sound_speed(a=self.a,
                           nd=self.nd,
                           kBT=self.kBT,
                           rc=self.rc,
                           mass=self.mass)

    def self_diffusion(self) -> float:
        return compute_self_diffusion(nd=self.nd,
                                      kBT=self.kBT,
                                      kpow=self.kpow,
                                      gamma=self.gamma,
                                      rc=self.rc)

    def dynamic_viscosity(self) -> float:
        return compute_dynamic_viscosity(nd=self.nd,
                                         mass=self.mass,
                                         kBT=self.kBT,
                                         kpow=self.kpow,
                                         gamma=self.gamma,
                                         rc=self.rc)

    def kinematic_viscosity(self) -> float:
        return compute_kinematic_viscosity(nd=self.nd,
                                           mass=self.mass,
                                           kBT=self.kBT,
                                           kpow=self.kpow,
                                           gamma=self.gamma,
                                           rc=self.rc)

    def MachNumber(self, U: float) -> float:
        """Compute the Mach number of the fluid for a given problem.
        Arguments:
            U: The typical velocity of the problem
        """
        return U / self.sound_speed()

    def ReynoldsNumber(self, U: float, L: float) -> float:
        """Compute the Reynolds number of the fluid for a given problem.
        Arguments:
            U: The typical velocity of the problem.
            L: The typical length of the problem.
        """
        return U * L / self.kinematic_viscosity()

    def SchmidtNumber(self) -> float:
        """ Compute the Schmidt number of the fluid."""
        return self.kinematic_viscosity() / self.self_diffusion()

    def interparticle_distance(self) -> float:
        """ Rough estimate of the inter-particle distance, used to estimate max dt. """
        nd = self.nd
        return 1 / nd**(1/3)

    def get_max_dt(self) -> float:
        """ Estimate maximum time step from empirical law as in Morris 1997."""

        h = self.interparticle_distance()
        dt_cfl = 0.25 * h / self.sound_speed()
        dt_visc = 0.125 * h**2 / self.kinematic_viscosity()
        return min((dt_cfl, dt_visc))


def create_dpd_params_from_str(s: str) -> DPDParams:
    import re
    rexf = '[-+]?\d*\.\d+|\d+'
    matches = re.findall(f"a_({rexf})_gamma_({rexf})_kBT_({rexf})_nd_({rexf})_mass_({rexf})_rc_({rexf})_kpow_({rexf})", s)
    assert len(matches) == 1
    values = [float(v) for v in matches[0]]
    return DPDParams(*values)


def _check_accuracy(varname: str,
                    val: float,
                    ref: float,
                    tol: float=1e-6):
    """ Issue a warning if the value is far (in relative error) from its target.

    Arguments:
        val: The value to check.
        ref: The target value.
        tol: The minimum relative error that triggers a warning.
    """
    err = abs(val - ref) / ref
    if err > tol:
        warnings.warn(f"Could not get {varname} to be {ref}. Got {val} ({err} relative error)", RuntimeWarning, stacklevel=3)


def create_dpd_params_from_Re_Ma(Re: float,
                                 Ma: float,
                                 U: float,
                                 L: float,
                                 *,
                                 a_rc_kBTinv: float=30,
                                 kpow: float=0.125,
                                 nd: float=10,
                                 rc: float=1,
                                 mass: float=1) -> DPDParams:

    from scipy.optimize import fsolve

    def equations(x):
        a = abs(x[0])
        gamma = abs(x[1])
        kBT = a * rc / a_rc_kBTinv
        p = DPDParams(a, gamma, kBT, nd, mass, rc, kpow)
        return (p.MachNumber(U) - Ma,
                p.ReynoldsNumber(U, L) - Re)

    x = fsolve(equations, (1, 1))
    a = abs(x[0])
    gamma = abs(x[1])
    kBT = a * rc / a_rc_kBTinv

    params = DPDParams(a, gamma, kBT, nd, mass, rc, kpow)
    _check_accuracy("Re", val=params.ReynoldsNumber(U, L), ref=Re)
    _check_accuracy("Ma", val=params.MachNumber(U), ref=Ma)
    return params

def create_dpd_params_from_props(kinematic_viscosity: float,
                                 sound_speed: float,
                                 *,
                                 a_rc_kBTinv: float=30,
                                 kpow: float=0.125,
                                 nd: float=10,
                                 rc: float=1,
                                 mass: float=1) -> DPDParams:

    from scipy.optimize import fsolve

    def equations(x):
        a = abs(x[0])
        gamma = abs(x[1])
        kBT = a * rc / a_rc_kBTinv
        p = DPDParams(a, gamma, kBT, nd, mass, rc, kpow)
        return (p.kinematic_viscosity() - kinematic_viscosity,
                p.sound_speed() - sound_speed)

    x = fsolve(equations, (1, 1))
    a = abs(x[0])
    gamma = abs(x[1])
    kBT = a * rc / a_rc_kBTinv

    params = DPDParams(a, gamma, kBT, nd, mass, rc, kpow)
    _check_accuracy("kinematic viscosity", val=params.kinematic_viscosity(), ref=kinematic_viscosity)
    _check_accuracy("sound speed", val=params.sound_speed(), ref=sound_speed)
    return params
