# Copyright 2020 ETH Zurich. All Rights Reserved.

import math
import sys
import unittest

sys.path.insert(0, '..')
from dpdprops import compute_kinematic_viscosity, compute_dynamic_viscosity

class TestViscosity(unittest.TestCase):
    def test_groot_warren(self):

        # arbitrary parameters
        gamma = 12.534
        nd = 8.345
        m = 0.2345
        kBT = 0.5123
        rc = 1.1231

        # standard DPD
        s = 2
        kpow = s/2

        # result from Groot and Warren for standard DPD
        nuK = 45 * kBT / (2 * math.pi * gamma * nd * rc**3) / 2
        nuD = 2 * math.pi * gamma * nd / m * rc**5 / 1575
        nuref = nuK + nuD

        nu = compute_kinematic_viscosity(kBT=kBT, gamma=gamma, nd=nd, mass=m, rc=rc, kpow=kpow)

        self.assertEqual(nu, nuref)

    def test_dynamic_viscosity(self):

        # arbitrary parameters
        gamma = 12.534
        nd = 8.345
        m = 0.2345
        kBT = 0.5123
        rc = 1.1231
        kpow = 0.125

        nu = compute_kinematic_viscosity(kBT=kBT, gamma=gamma, nd=nd, mass=m, rc=rc, kpow=kpow)
        eta = compute_dynamic_viscosity(kBT=kBT, gamma=gamma, nd=nd, mass=m, rc=rc, kpow=kpow)

        self.assertEqual(nu * nd * m, eta)

    def test_dimensions(self):
        from pint import UnitRegistry
        ureg = UnitRegistry()

        gamma = 12.5 * ureg.gram / ureg.second
        nd = 11 * ureg.meter**(-3)
        m = 1.0 * ureg.gram
        kBT = 1.0 * ureg.joule
        rc = 1.0 * ureg.meter
        kpow = 1/2

        nu = compute_kinematic_viscosity(kBT=kBT, gamma=gamma, nd=nd, mass=m, rc=rc, kpow=kpow)
        eta = compute_dynamic_viscosity(kBT=kBT, gamma=gamma, nd=nd, mass=m, rc=rc, kpow=kpow)

        self.assertTrue(nu.check('[length]**2 / [time]'))
        self.assertTrue(eta.check('[viscosity]'))
