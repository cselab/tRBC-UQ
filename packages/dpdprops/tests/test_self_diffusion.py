# Copyright 2020 ETH Zurich. All Rights Reserved.

import math
import sys
import unittest

sys.path.insert(0, '..')
from dpdprops import compute_self_diffusion

class TestSelfDiffusion(unittest.TestCase):
    def test_groot_warren(self):

        # arbitrary parameters
        gamma = 12.534
        nd = 8.345
        kBT = 0.5123
        rc = 1.1231

        # standard DPD
        s = 2
        kpow = s/2

        # result from Groot and Warren for standard DPD
        Dref = 45 * kBT / (2 * math.pi * gamma * nd * rc**3)
        D = compute_self_diffusion(kBT=kBT, gamma=gamma, nd=nd, rc=rc, kpow=kpow)

        self.assertEqual(D, Dref)

    def test_dimensions(self):
        from pint import UnitRegistry
        ureg = UnitRegistry()

        gamma = 12.5 * ureg.gram / ureg.second
        nd = 11 * ureg.meter**(-3)
        kBT = 1.0 * ureg.joule
        rc = 1.0 * ureg.meter
        kpow = 1/2

        D = compute_self_diffusion(gamma=gamma, nd=nd, kBT=kBT, rc=rc, kpow=kpow)
        self.assertTrue(D.check('[length]**2 / [time]'))
