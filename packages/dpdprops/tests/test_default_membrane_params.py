# Copyright 2020 ETH Zurich. All Rights Reserved.

import os
import sys
import unittest

sys.path.insert(0, '..')
from dpdprops import (KantorWLCRBCDefaultParams,
                      JuelicherLimRBCDefaultParams)

here = os.path.dirname(os.path.realpath(__file__))

class TestMembraneParams(unittest.TestCase):
    def test_KantorWLCRBCDefaultParams(self):
        from pint import UnitRegistry
        import trimesh

        ureg = UnitRegistry()

        with self.assertRaises(ValueError):
            params = KantorWLCRBCDefaultParams(ureg, D0=1*ureg.J)

        params = KantorWLCRBCDefaultParams(ureg)

        length_scale = 1 * ureg.um
        time_scale = 1 * ureg.us
        mass_scale = 1 * ureg.pg

        mesh_file = os.path.join(here, "data", "rbc_mesh.off")
        mesh = trimesh.load_mesh(mesh_file, process=False)

        prms = params.get_params(length_scale=length_scale,
                                 time_scale=time_scale,
                                 mass_scale=mass_scale,
                                 mesh=mesh)

    def test_JuelicherLimRBCDefaultParams(self):
        from pint import UnitRegistry
        import trimesh

        ureg = UnitRegistry()

        with self.assertRaises(ValueError):
            params = JuelicherLimRBCDefaultParams(ureg, D0=1*ureg.J)

        params = JuelicherLimRBCDefaultParams(ureg)

        length_scale = 1 * ureg.um
        time_scale = 1 * ureg.us
        mass_scale = 1 * ureg.pg

        mesh_file = os.path.join(here, "data", "rbc_mesh.off")
        mesh = trimesh.load_mesh(mesh_file, process=False)

        prms = params.get_params(length_scale=length_scale,
                                 time_scale=time_scale,
                                 mass_scale=mass_scale,
                                 mesh=mesh)

        self.assertLess(params.get_reduced_volume(), 1)
