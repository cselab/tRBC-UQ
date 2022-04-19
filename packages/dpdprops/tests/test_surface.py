# Copyright 2020 ETH Zurich. All Rights Reserved.

import math
import os
import sys
import unittest

sys.path.insert(0, '..')
from dpdprops import (average_intervertex_distance,
                      equivalent_sphere_radius,
                      reduced_volume)

here = os.path.dirname(os.path.realpath(__file__))

def compute_mean_edge_dist(mesh):
    import numpy as np
    vertices = np.array(mesh.vertices)
    faces    = np.array(mesh.faces)

    def dist(v, ids1, ids2):
        d = np.sqrt(np.sum((v[ids1,:] - v[ids2,:])**2, axis=1))
        return np.mean(d)

    d0 = dist(vertices, faces[:,0], faces[:,1])
    d1 = dist(vertices, faces[:,0], faces[:,2])
    d2 = dist(vertices, faces[:,2], faces[:,1])
    return np.mean([d0, d1, d2])

class TestSurface(unittest.TestCase):

    def load_example_mesh(self):
        import trimesh
        mesh_file = os.path.join(here, "data", "rbc_mesh.off")
        return trimesh.load_mesh(mesh_file, process=False)

    def test_intervertex_distance(self):
        mesh = self.load_example_mesh()
        area = mesh.area
        nv = len(mesh.vertices)
        l = average_intervertex_distance(area=area, nv=nv)
        self.assertAlmostEqual(l,
                               compute_mean_edge_dist(mesh),
                               places=2)

    def test_equivalent_radius(self):
        R = 42

        # area of a sphere: the equivalent radius should be R
        area = 4 * math.pi * R**2
        Req = equivalent_sphere_radius(area=area)
        self.assertEqual(R, Req)

    def test_reduced_volume(self):
        R = 42

        # area and volume of a sphere: the reduced volume should be 1
        area = 4 * math.pi * R**2
        volume = 4 * math.pi / 3 * R**3

        v = reduced_volume(area=area, volume=volume)
        self.assertEqual(v, 1)


    def test_dimensions(self):
        from pint import UnitRegistry
        ureg = UnitRegistry()

        A = 50 * ureg.m**2
        V = 450 * ureg.m**3
        nv = 123

        l = average_intervertex_distance(area=A, nv=nv)
        RA = equivalent_sphere_radius(area=A)
        v = reduced_volume(area=A, volume=V)

        self.assertTrue(l.check('[length]'))
        self.assertTrue(RA.check('[length]'))
        self.assertTrue(v.check('[]'))
