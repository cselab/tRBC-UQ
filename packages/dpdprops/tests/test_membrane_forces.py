# Copyright 2020 ETH Zurich. All Rights Reserved.

import numpy as np
import sys
import trimesh
import unittest

sys.path.insert(0, '..')
from dpdprops import (KantorParams,
                      JuelicherParams,
                      extract_dihedrals,
                      compute_kantor_energy,
                      compute_juelicher_energy)

class TestMembraneForces(unittest.TestCase):

    def test_kantor(self):

        R = 1.0
        kb = 1 * np.sqrt(3)/2

        mesh = trimesh.creation.icosphere(radius=R, subdivisions=3)
        dihedrals = extract_dihedrals(faces=mesh.faces)

        params = KantorParams(kb=kb)

        E = compute_kantor_energy(vertices=mesh.vertices,
                                  dihedrals=dihedrals,
                                  params=params)

        self.assertAlmostEqual(E, 25.978990763)

        with self.assertRaises(NotImplementedError):
            compute_kantor_energy(vertices=mesh.vertices,
                                  dihedrals=dihedrals,
                                  params=KantorParams(kb=kb, theta=1))



    def test_juelicher(self):

        R = 1.0
        kb = 1

        mesh = trimesh.creation.icosphere(radius=R, subdivisions=3)
        dihedrals = extract_dihedrals(faces=mesh.faces)

        params = JuelicherParams(kb=kb)

        E = compute_juelicher_energy(vertices=mesh.vertices,
                                     faces=mesh.faces,
                                     dihedrals=dihedrals,
                                     params=params)

        self.assertAlmostEqual(E, 8*np.pi, places=1)

        E = compute_juelicher_energy(vertices=mesh.vertices,
                                     faces=mesh.faces,
                                     dihedrals=dihedrals,
                                     params=JuelicherParams(kb=kb, C0=2/R))

        self.assertAlmostEqual(E, 0, places=1)

        with self.assertRaises(NotImplementedError):
            compute_juelicher_energy(vertices=mesh.vertices,
                                     faces=mesh.faces,
                                     dihedrals=dihedrals,
                                     params=JuelicherParams(kb=kb, kad=1))
