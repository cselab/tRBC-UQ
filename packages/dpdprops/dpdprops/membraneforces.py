#!/usr/bin/env python

import numpy as np

from .membraneparams import (KantorParams,
                             JuelicherParams,
                             WLCParams,
                             LimParams,
                             MembraneParams)

def extract_dihedrals(*,
                      faces: np.ndarray) -> np.ndarray:

    """
    Find dihedrals from face connectivity information.
    Assume a closed triangle mesh.
    The order of the indices are as follow:

    (b, c, a, d)

        a
       /|\
      / | \
    d   |  c
      \ | /
       \|/
        b

    Arguments:
        faces: Face connectivity of the triangle mesh.

    Returns:
        The dihedral informtions (one entry of 4 indices per dihedral)
    """

    edges_to_faces = {}

    for faceid, f in enumerate(faces):
        for i in range(3):
            edge = (f[i], f[(i+1)%3])
            edges_to_faces[edge] = faceid

    dihedrals = []

    for faceid, f0 in enumerate(faces):
        for i in range(3):
            a = f0[i]
            b = f0[(i+1)%3]
            c = f0[(i+2)%3]
            edge = (b, a)
            otherfaceid = edges_to_faces[edge]
            f1 = faces[otherfaceid]
            d = [v for v in f1 if v != a and v != b]
            assert len(d) == 1
            d = d[0]
            dihedrals.append([b, c, a, d])

    return np.array(dihedrals)


def _compute_dihedral_normals(dihedrals, vertices):
    b = vertices[dihedrals[:,0],:]
    c = vertices[dihedrals[:,1],:]
    a = vertices[dihedrals[:,2],:]
    d = vertices[dihedrals[:,3],:]
    ab = b-a
    n0 = np.cross(ab, c-a)
    n1 = np.cross(ab, a-d)
    n0 /= np.linalg.norm(n0, axis=1)[:,np.newaxis]
    n1 /= np.linalg.norm(n1, axis=1)[:,np.newaxis]
    return n0, n1

def _compute_triangle_areas(faces, vertices):
    a = vertices[faces[:,0],:]
    b = vertices[faces[:,1],:]
    c = vertices[faces[:,2],:]
    n = np.cross(b-a, c-a)
    return 0.5 * np.linalg.norm(n, axis=1)


def compute_kantor_energy(*,
                          vertices: np.ndarray,
                          dihedrals: np.ndarray,
                          params: "KantorParams") -> float:
    """
    Compute the bending energy using the Kantor-Nelson model.
    Assumes that theta0 = 0.

    Arguments:
        vertices: positions of the mesh.
        dihedrals: list of dihedrals of the mesh.
        params: Kantor parameters.

    Return:
        The bending energy of the mesh.
    """
    kb = params.kb
    theta0 = params.theta
    if theta0 != 0:
        raise NotImplementedError(f"Nelson energy: Not implemented for theta0 != 0.")

    n0, n1 = _compute_dihedral_normals(dihedrals, vertices)

    E = 2 * kb * np.sum((1 - np.sum(n0*n1, axis=1)))
    return E


def compute_juelicher_energy(*,
                             vertices: np.ndarray,
                             faces: np.ndarray,
                             dihedrals: np.ndarray,
                             params: "JuelicherParams") -> float:
    """
    Compute the bending energy using the Juelicher model.

    Arguments:
        vertices: positions of the mesh.
        faces: connectivity of the mesh.
        params: Juelicher parameters.

    Return:
        The bending energy of the mesh.
    """
    kb = params.kb
    H0 = params.C0

    if params.kad != 0 or params.DA0 != 0:
        raise NotImplementedError(f"Juelicher energy: Not implemented for kad != 0.")

    nv = len(vertices)

    vertex_areas = np.zeros(nv)
    faces_areas = _compute_triangle_areas(faces, vertices)
    np.add.at(vertex_areas, faces[:,0], faces_areas)
    np.add.at(vertex_areas, faces[:,1], faces_areas)
    np.add.at(vertex_areas, faces[:,2], faces_areas)
    vertex_areas /= 3

    vertex_mean_curvatures = np.zeros(nv)

    b = vertices[dihedrals[:,0],:]
    c = vertices[dihedrals[:,1],:]
    a = vertices[dihedrals[:,2],:]
    d = vertices[dihedrals[:,3],:]

    ab = b-a
    n0 = np.cross(ab, c-a)
    n1 = np.cross(ab, a-d)

    arg = np.sum(n0*n1, axis=1) / (np.linalg.norm(n0, axis=1)*np.linalg.norm(n1, axis=1))
    theta = np.arccos(np.maximum(-1, np.minimum(1, arg)))
    l = np.linalg.norm(ab, axis=1)
    ltheta = l * theta

    np.add.at(vertex_mean_curvatures, dihedrals[:,2], ltheta)
    np.add.at(vertex_mean_curvatures, dihedrals[:,0], ltheta)

    vertex_mean_curvatures /= 4 * vertex_areas

    E = 0.5 * kb * np.sum((vertex_mean_curvatures - H0)**2 * vertex_areas)
    return E
