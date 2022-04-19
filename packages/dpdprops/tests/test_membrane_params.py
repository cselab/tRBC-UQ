# Copyright 2020 ETH Zurich. All Rights Reserved.

import numpy as np
import sys
import unittest

sys.path.insert(0, '..')
from dpdprops import (MembraneParams,
                      KantorParams,
                      JuelicherParams,
                      WLCParams,
                      LimParams,
                      average_intervertex_distance,
                      kantor_to_bending_modulus,
                      shear_modulus_to_wlc)

class TestMembraneParams(unittest.TestCase):
    def test_l0(self):
        area = 45.0
        nv = 20
        p = MembraneParams(area=area, nv=nv, volume=0, ka=0, kv=0, gamma=0, bending_params=None, shear_params=None)
        self.assertEqual(p.get_l0(),
                         average_intervertex_distance(area=area, nv=nv))

    def test_kantor(self):
        kb = 123
        p = MembraneParams(area=0, nv=0, volume=0, ka=0, kv=0, gamma=0, bending_params=KantorParams(kb=kb), shear_params=None)
        self.assertEqual(kantor_to_bending_modulus(kb=kb), p.bending_modulus())

    def test_juelicher(self):
        kb = 123
        p = MembraneParams(area=0, nv=0, volume=0, ka=0, kv=0, gamma=0, bending_params=JuelicherParams(kb=kb), shear_params=None)
        self.assertEqual(kb, p.bending_modulus())

    def test_wlc(self):
        area = 45
        nv = 20

        mu = 123
        ka = 3
        x0 = 0.3
        l0 = average_intervertex_distance(area=area, nv=nv)
        m = 2
        ks = shear_modulus_to_wlc(mu=mu, x0=x0, l0=l0, m=m)
        p = MembraneParams(area=area, nv=nv, volume=0, ka=0, kv=0, gamma=0, bending_params=None, shear_params=WLCParams(ka=ka, ks=ks, x0=x0, m=m))
        self.assertAlmostEqual(mu, p.shear_modulus())

    def test_lim(self):
        mu = 1234
        ka = 3
        p = MembraneParams(area=0, nv=0, volume=0, ka=0, kv=0, gamma=0, bending_params=None, shear_params=LimParams(ka=ka, mu=mu))
        self.assertEqual(mu, p.shear_modulus())

    def test_to_interaction_wlc_kantor(self):
        area = 45
        nv = 20

        mu = 123
        ka = 3
        x0 = 0.3
        l0 = average_intervertex_distance(area=area, nv=nv)
        m = 2
        ks = shear_modulus_to_wlc(mu=mu, x0=x0, l0=l0, m=m)

        kb = 123

        p = MembraneParams(area=area, nv=nv, volume=0, ka=0, kv=0, gamma=0,
                           bending_params=KantorParams(kb=kb),
                           shear_params=WLCParams(ka=ka, ks=ks, x0=x0, m=m))

        d = p.to_interactions()

        self.assertEqual(d['shear_desc'], 'wlc')
        self.assertEqual(d['bending_desc'], 'Kantor')

    def test_to_interaction_lim_juelicher(self):
        area = 45
        nv = 20

        mu = 123
        ka = 3
        kb = 123

        p = MembraneParams(area=area, nv=nv, volume=0, ka=0, kv=0, gamma=0,
                           bending_params=JuelicherParams(kb=kb),
                           shear_params=LimParams(ka=ka, mu=mu))

        d = p.to_interactions()

        self.assertEqual(d['shear_desc'], 'Lim')
        self.assertEqual(d['bending_desc'], 'Juelicher')

    def test_to_interaction_visc(self):
        area = 45
        nv = 20

        mu = 123
        ka = 3
        kb = 123
        gamma = 23
        kBT = 2

        p = MembraneParams(area=area, nv=nv, volume=0, ka=0, kv=0, gamma=gamma, kBT=kBT,
                           bending_params=JuelicherParams(kb=kb),
                           shear_params=LimParams(ka=ka, mu=mu))

        d = p.to_interactions()
        self.assertEqual(d['gammaC'], gamma)
        self.assertEqual(d['kBT'], kBT)

        d = p.to_interactions_zero_visc()
        self.assertEqual(d['gammaC'], 0)
        self.assertEqual(d['kBT'], 0)

        d = p.to_viscous()
        self.assertEqual(d['gammaC'], gamma)
        self.assertEqual(d['kBT'], kBT)


    def test_visc(self):
        area = 45
        nv = 20

        mu = 123
        ka = 3
        kb = 123

        p = MembraneParams(area=area, nv=nv, volume=0, ka=0, kv=0, gamma=0,
                           bending_params=JuelicherParams(kb=kb),
                           shear_params=LimParams(ka=ka, mu=mu))

        etam = 42
        p.set_viscosity(etam)
        self.assertEqual(p.get_viscosity(), etam)


    def test_dt_max_units(self):
        from pint import UnitRegistry

        ureg = UnitRegistry()
        area = 45 * ureg.m**2
        volume = 45 * ureg.m**3
        nv = 20

        mu = 123 * ureg.N / ureg.m
        ka = 3 * ureg.N / ureg.m

        kA = 3 * ureg.J / ureg.m**2
        kV = 3 * ureg.J / ureg.m**3
        kb = 123 * ureg.J
        gamma = 1 * ureg.Pa * ureg.s * ureg.m

        mass = 1.0 * ureg.kg

        p = MembraneParams(area=area, nv=nv, volume=volume, ka=kA, kv=kV, gamma=gamma,
                           bending_params=JuelicherParams(kb=kb),
                           shear_params=LimParams(ka=ka, mu=mu))

        dt = p.get_max_dt(mass=mass)
        self.assertTrue(dt.check('[time]'))

    def test_dt_max_zero_visc(self):
        area = 45
        volume = 45
        nv = 20

        mu = 123
        ka = 3

        kA = 3
        kV = 3
        kb = 123
        gamma = 0 # zero visc

        mass = 1.0

        p = MembraneParams(area=area, nv=nv, volume=volume, ka=kA, kv=kV, gamma=gamma,
                           bending_params=JuelicherParams(kb=kb),
                           shear_params=LimParams(ka=ka, mu=mu))

        dt = p.get_max_dt_visc(mass=mass)
        self.assertFalse(np.isnan(dt))
        self.assertFalse(np.isinf(dt))
        self.assertGreater(dt, 0)
