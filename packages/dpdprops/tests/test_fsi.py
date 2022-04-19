# Copyright 2020 ETH Zurich. All Rights Reserved.

import math
import sys
import unittest

sys.path.insert(0, '..')
from dpdprops import (get_gamma_fsi_DPD_membrane,
                      DPDParams,
                      create_fsi_dpd_params)

class TestFSI(unittest.TestCase):

    def test_dimensions(self):
        from pint import UnitRegistry
        ureg = UnitRegistry()

        eta = 1.0 * ureg.Pa * ureg.s
        kpow = 0.125
        nd_membrane = 5 / ureg.um**2
        rc = 1.0 * ureg.um
        nd_fluid = 10 / rc**3

        gamma = get_gamma_fsi_DPD_membrane(eta=eta, kpow=kpow, nd_membrane=nd_membrane, nd_fluid=nd_fluid, rc=rc)

        self.assertTrue(gamma.check('[force]/[velocity]')) # FD ~ gamma * vij

    def test_fsi_dpd_params(self):

        fluid_params = DPDParams(a=25,
                                 gamma=30,
                                 kBT=1,
                                 nd=10,
                                 mass=1,
                                 rc=1,
                                 kpow=0.125)

        nd_membrane = 5

        fsi_params = create_fsi_dpd_params(fluid_params=fluid_params,
                                           nd_membrane=nd_membrane)

        self.assertEqual(fsi_params.a, 0)
        self.assertEqual(fsi_params.gamma,
                         get_gamma_fsi_DPD_membrane(eta=fluid_params.dynamic_viscosity(),
                                                    kpow=fluid_params.kpow,
                                                    nd_membrane=nd_membrane,
                                                    nd_fluid=fluid_params.nd,
                                                    rc=fluid_params.rc))
