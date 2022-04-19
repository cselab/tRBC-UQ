#!/usr/bin/env python

from dataclasses import dataclass, asdict
import math

from .membrane import *

class BendingParams:
    pass

class ShearParams:
    pass

@dataclass
class KantorParams(BendingParams):
    """
    Bending parameters for the Kantor model:
        kb: The bending energy coefficient.
        theta: The equilibrium angle between two adjacent triangles.
    """
    kb: float
    theta: float = 0

    def to_interactions(self) -> dict:
        """
        get a dictionary that has the parameters as in Mirheo
        https://mirheo.readthedocs.io/en/latest/user/interactions.html#mirheo.Interactions.MembraneForces.__init__
        """
        return {"bending_desc": "Kantor",
                "kb": self.kb,
                "theta": self.theta}

    def bending_modulus(self) -> float:
        """
        Returns:
            The macroscopic bending modulus.
        """
        return kantor_to_bending_modulus(kb=self.kb)


@dataclass
class JuelicherParams(BendingParams):
    """
    Bending parameters for the Juelicher model:
        kb: The bending modulus.
        C0: Spontaneous mean curvature
        kad coefficient for area difference model
        DA0: area difference.
    """
    kb: float
    C0: float = 0
    kad: float = 0
    DA0: float = 0

    def to_interactions(self) -> dict:
        """
        get a dictionary that has the parameters as in Mirheo
        https://mirheo.readthedocs.io/en/latest/user/interactions.html#mirheo.Interactions.MembraneForces.__init__
        """
        return {"bending_desc": "Juelicher",
                "kb": self.kb,
                "C0": self.C0,
                "kad": self.kad,
                "DA0": self.DA0}

    def bending_modulus(self) -> float:
        """
        Returns:
            The macroscopic bending modulus.
        """
        return self.kb


@dataclass
class WLCParams(ShearParams):
    """
    Shear parameters for the WLC model:
        ka: Local area energy coefficient.
        ks: The spring force coefficient.
        x0: l0/lmax.
        m Exponent of the chain.
    """
    ka: float
    ks: float
    x0: float
    m: float = 2

    def to_interactions(self) -> dict:
        """
        get a dictionary that has the parameters as in Mirheo
        https://mirheo.readthedocs.io/en/latest/user/interactions.html#mirheo.Interactions.MembraneForces.__init__
        """
        return {"shear_desc": "wlc",
                "ks": self.ks,
                "x0": self.x0,
                "mpow": self.m,
                "ka": self.ka}

    def shear_modulus(self, l0: float) -> float:
        return wlc_to_shear_modulus(ks=self.ks, l0=l0, x0=self.x0, m=self.m)


@dataclass
class LimParams(ShearParams):
    """
    Shear parameters for the Lim model.
    """

    ka: float
    mu: float
    a3: float = -2
    a4: float = 8
    b1: float = 0.7
    b2: float = 0.75

    def to_interactions(self) -> dict:
        """
        get a dictionary that has the parameters as in Mirheo
        https://mirheo.readthedocs.io/en/latest/user/interactions.html#mirheo.Interactions.MembraneForces.__init__
        """
        return {"shear_desc": "Lim",
                "ka": self.ka,
                "mu": self.mu,
                "a3": self.a3,
                "a4": self.a4,
                "b1": self.b1,
                "b2": self.b2}

    def shear_modulus(self, l0: float) -> float:
        return self.mu

@dataclass
class MembraneParams:
    """
    Membrane parameters class.
    Holds all parameters of an elastic membrane.
    """
    area: float
    volume: float
    ka: float
    kv: float
    nv: int
    bending_params: BendingParams
    shear_params: ShearParams
    gamma: float = 0
    kBT: float = 0

    def get_l0(self) -> float:
        """
        Compute the average length of the edges.
        """
        return average_intervertex_distance(area=self.area,
                                            nv=self.nv)

    def bending_modulus(self) -> float:
        return self.bending_params.bending_modulus()

    def shear_modulus(self) -> float:
        l0 = self.get_l0()
        return self.shear_params.shear_modulus(l0)

    def get_viscosity(self) -> float:
        return membrane_gamma_to_membrane_viscosity(gC=self.gamma)

    def set_viscosity(self, etam: float) -> float:
        self.gamma = membrane_viscosity_to_membrane_gamma(etam=etam)

    def to_interactions_zero_visc(self) -> dict:
        """
        get a dictionary that has the parameters as in Mirheo, with gammaC and kBT set to 0
        https://mirheo.readthedocs.io/en/latest/user/interactions.html#mirheo.Interactions.MembraneForces.__init__
        """
        return {"tot_area": self.area,
                "tot_volume": self.volume,
                "ka_tot": self.ka,
                "kv_tot": self.kv,
                "kBT": 0,
                "gammaC": 0,
                **self.bending_params.to_interactions(),
                **self.shear_params.to_interactions()}

    def to_viscous(self) -> dict:
        """
        get a dictionary that has the viscous and fluctuation parameters as in Mirheo
        https://mirheo.readthedocs.io/en/latest/user/interactions.html#mirheo.Interactions.MembraneForces.__init__
        """
        return {"gammaC": self.gamma,
                "kBT": self.kBT}

    def to_interactions(self) -> dict:
        """
        get a dictionary that has the parameters as in Mirheo
        https://mirheo.readthedocs.io/en/latest/user/interactions.html#mirheo.Interactions.MembraneForces.__init__
        """
        res = self.to_interactions_zero_visc()
        visc = self.to_viscous()
        for key in visc.keys():
            res[key] = visc[key]
        return res


    def get_max_dt_visc(self, *, mass) -> float:
        """
        Estimate the maximum time step allowed for the viscous part.

        Arguments:
            mass: The mass of one particle.

        Return:
            dt: The maximum timestep estimate.
        """
        h = self.get_l0()
        rho = self.nv * mass / self.area
        nu = self.get_viscosity() / rho

        # handle zero viscosity case for floats or pint quantities
        if isinstance(nu, float):
            nu = max([nu, 1e-6])
        else: # pint
            nu = max([nu, 1e-6 * nu.units])
        dt_visc = 0.125 * h**2 / nu
        return dt_visc


    def get_max_dt_elastic(self, *, mass) -> float:
        """
        Estimate the maximum time step allowed for the elastic part.

        Arguments:
            mass: The mass of one particle.

        Return:
            dt: The maximum timestep estimate.
        """
        h = self.get_l0()
        f_el = self.shear_modulus() * h + self.bending_modulus() / h + self.ka * h + self.kv * h**2
        acc_el = f_el / mass
        dt_el = (h / acc_el)**(1/2)
        return dt_el


    def get_max_dt(self, *, mass) -> float:
        """
        Estimate the maximum time step.

        Arguments:
            mass: The mass of one particle

        Return:
            dt: The maximum timestep estimate.
        """
        dt_visc = self.get_max_dt_visc(mass=mass)
        dt_el = self.get_max_dt_elastic(mass=mass)

        return min([dt_visc, dt_el])


def _default_if_not_None(val, default):
    if val is None:
        return default
    else:
        return val

def _check_units(var, units: str):
    if not var.check(units):
        raise ValueError(f"wrong units: expected {units}, got {var}")


class DefaultRBCParams:
    """
    A class to wrap the default physical parameters of a Red Blood Cell (RBC).
    The parameters are taken from:
        Lim, H. W. G., M. Wortis, and R. Mukhopadhyay.
        "Red blood cell shapes and shape transformations: Newtonian mechanics of a composite membrane."
        Soft matter 4 (2008): 83-249.
    """

    def __init__(self, ureg,
                 *,
                 ka = None,
                 mu = None,
                 kappab = None,
                 kA = None,
                 kV = None,
                 D0 = None,
                 A0 = None,
                 V0 = None,
                 eta_m = None,
                 kBT = None):
        """
        Arguments:
            ureg: A pint unit registry.
            ka: Compression modulus
            mu: Shear modulus
            kabbab: Bending modulus.
            kA: Area constraint coefficient (energy per area)
            kV: Volume constraint coefficient (energy per volume)
            D0: Diameter of the RBC at rest.
            A0: Area of the RBC at rest.
            V0: Volume of the RBC at rest.
            eta_m: (2D) Membrane viscosity.
            kBT: The membrane temperature (in energy units).
        """

        self.ureg = ureg

        kB = 1.38064852e-23 * ureg.J / ureg.K
        T = ureg.Quantity(20, ureg.degC)

        self.ka = _default_if_not_None(ka, 4.99 * ureg.uN / ureg.m)
        self.mu = _default_if_not_None(mu, 4.99 * ureg.uN / ureg.m)
        self.kappab = _default_if_not_None(kappab, 2.10e-19 * ureg.J)
        self.kA = _default_if_not_None(kA, 0.5 * ureg.J / ureg.m**2)
        self.kV = _default_if_not_None(kV, 7.23e5 * ureg.J / ureg.m**3)

        # geometric parameters from Evans & Fung 1972
        self.D0 = _default_if_not_None(D0, 7.82 * ureg.um)
        self.A0 = _default_if_not_None(A0, 135 * ureg.um**2)
        self.V0 = _default_if_not_None(V0, 94  * ureg.um**3)

        self.eta_m = _default_if_not_None(eta_m, 0.42e-6 * ureg.Pa * ureg.s * ureg.m)
        self.kBT = _default_if_not_None(kBT, kB * T.to('K'))

        _check_units(self.ka,     '[force] / [length]')
        _check_units(self.mu,     '[force] / [length]')
        _check_units(self.kappab, '[energy]')
        _check_units(self.kA,     '[energy] / [area]')
        _check_units(self.kV,     '[energy] / [volume]')
        _check_units(self.D0,     '[length]')
        _check_units(self.A0,     '[area]')
        _check_units(self.V0,     '[volume]')
        _check_units(self.eta_m,  '[viscosity] * [length]')
        _check_units(self.kBT,    '[energy]')


    def get_reduced_volume(self):
        R = equivalent_sphere_radius(area=self.A0)
        V = 4 * math.pi / 3 * R**3
        return float(self.V0/V)



class KantorWLCRBCDefaultParams(DefaultRBCParams):
    """
    Wrapper class for defualt physical parameters and converts it to Kantor + WLC model parameters in simulation units.
    """

    def __init__(self, ureg, *,
                 ka=None,
                 mu=None,
                 kappab=None,
                 kA=None,
                 kV=None,
                 D0=None,
                 A0=None,
                 V0=None,
                 eta_m=None,
                 kBT=None,
                 x0=None):
        """
        Arguments:
            See DefaultRBCParams.
            x0: l0/lmax.
        """

        mu = _default_if_not_None(mu, 4.058 * ureg.uN / ureg.m)
        ka = mu
        kappab = _default_if_not_None(kappab, 5.906e-19 * ureg.J)

        super().__init__(ureg, ka=ka, mu=mu, kappab=kappab, kA=kA, kV=kV, D0=D0, A0=A0, V0=V0, eta_m=eta_m, kBT=kBT)
        self.x0 = _default_if_not_None(x0, 0.416) # from UQ paper


    def get_params(self, *, length_scale, time_scale, mass_scale, mesh):
        force_scale = length_scale * mass_scale / time_scale**2

        mu = float(self.mu * length_scale / force_scale)
        kappab = float(self.kappab / force_scale / length_scale)
        ka = float(self.ka * length_scale / force_scale)

        kA = float(self.kA * length_scale / force_scale)
        kV = float(self.kV * length_scale**2 / force_scale)

        ks = shear_modulus_to_wlc(mu=mu,
                                  l0=average_intervertex_distance(area=mesh.area, nv=len(mesh.vertices)),
                                  x0=self.x0)

        kb = bending_modulus_to_kantor(kc=kappab)

        A0 = float(self.A0 / length_scale**2)
        V0 = float(self.V0 / length_scale**3)

        etam = float(self.eta_m * time_scale / mass_scale)
        gamma = membrane_viscosity_to_membrane_gamma(etam=etam)

        kBT = float(self.kBT * time_scale**2 / length_scale**2 / mass_scale)

        return MembraneParams(area=A0,
                              volume=V0,
                              ka=kA,
                              kv=kV,
                              gamma=gamma,
                              kBT=kBT,
                              nv=len(mesh.vertices),
                              bending_params=KantorParams(kb=kb),
                              shear_params=WLCParams(ka=ka, ks=ks, x0=self.x0))



class JuelicherLimRBCDefaultParams(DefaultRBCParams):
    """
    Wrapper class for defualt physical parameters and converts it to Juelicher + Lim model parameters in simulation units.
    """

    def __init__(self, ureg, *,
                 ka=None,
                 mu=None,
                 kappab=None,
                 kA=None,
                 kV=None,
                 D0=None,
                 A0 = None,
                 V0 = None,
                 eta_m=None,
                 kBT=None,
                 delta=None,
                 m0_bar: float=0, # Default: no DA0; can be 10 as in Lim 2008, fig 2.29 AD(5) (= "Normal" RBC)
                 CADE: float=0,
                 a3: float=-2,
                 a4: float=8,
                 b1: float=0.7,
                 b2: float=1.84):
        """
        Arguments:
            See DefaultRBCParams.
            delta: Offset of leaflet midplanes.
            m0_bar: Non dimensional area difference parameter.
            CADE: ratio kADE / kappab. Set to 0 for no ADE contribution. Lim et al. used 2/pi.
            a3, a4, b1, b2: non linear coefficients (see Lim 2008)
        """

        super().__init__(ureg, ka=ka, mu=mu, kappab=kappab, kA=kA, kV=kV, D0=D0, A0=A0, V0=V0, eta_m=eta_m, kBT=kBT)
        self.delta = _default_if_not_None(delta, 2 * ureg.nm)
        self.m0_bar = m0_bar
        self.CADE = CADE
        self.a3 = a3
        self.a4 = a4
        self.b1 = b1
        self.b2 = b2


    def get_params(self, *, length_scale, time_scale, mass_scale, mesh):
        force_scale = length_scale * mass_scale / time_scale**2

        mu = float(self.mu * length_scale / force_scale)
        kappab = float(self.kappab / force_scale / length_scale)
        ka = float(self.ka * length_scale / force_scale)

        kA = float(self.kA * length_scale / force_scale)
        kV = float(self.kV * length_scale**2 / force_scale)

        kADE = self.CADE * kappab

        delta = float(self.delta / length_scale) # distance between 2 layers of RBC membrane

        A0 = float(self.A0 / length_scale**2)
        V0 = float(self.V0 / length_scale**3)
        RA = equivalent_sphere_radius(area=A0)

        DA0 = 2 * self.m0_bar * RA * delta

        etam = float(self.eta_m * time_scale / mass_scale)
        gamma = membrane_viscosity_to_membrane_gamma(etam=etam)

        kBT = float(self.kBT * time_scale**2 / length_scale**2 / mass_scale)

        return MembraneParams(area=A0,
                              volume=V0,
                              ka=kA,
                              kv=kV,
                              gamma=gamma,
                              kBT=kBT,
                              nv=len(mesh.vertices),
                              bending_params=JuelicherParams(kb=kappab, kad=kADE, DA0=DA0),
                              shear_params=LimParams(ka=ka, mu=mu, a3=self.a3, a4=self.a4, b1=self.b1, b2=self.b2))
