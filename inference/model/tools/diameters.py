#! /usr/bin/env python

import numpy as np
from .reorient import center_align_mesh

def compute_diameters(mesh):
    """
    Compute the diameters of the stretched cell.
    This corresponds to the two largest sides of the bounding box of the cell.

    Args:
        mesh: the mesh of the cell
    Return:
        Dlong, Dshort: the two diameters
    """
    mesh = center_align_mesh(mesh)
    extents = np.ptp(mesh.vertices, axis=0)
    extents = np.sort(extents)[::-1]

    Dlong  = extents[0]
    Dshort = extents[1]
    return Dlong, Dshort
