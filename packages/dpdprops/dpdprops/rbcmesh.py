#!/usr/bin/env python

import os
import trimesh

SCRIPT_PATH = os.path.dirname(os.path.realpath(__file__))
DATA_PATH = os.path.join(SCRIPT_PATH, 'data')
DEFAULT_STRESS_FREE_REDUCED_VOLUME = 0.95

def _load_mesh(mesh_name: str):
    """
    Load a mesh from a file using trimesh

    Arguments:
        mesh_name: the file name containing the triangle mesh.
    """
    return trimesh.load(mesh_name, process=False)


def load_stress_free_mesh(subdivisions: int=4,
                          reduced_volume: float=DEFAULT_STRESS_FREE_REDUCED_VOLUME):
    """
    Load the default stess-free mesh of a healthy RBC.

    Arguments:
        subdivisions: The refinement of the mesh.
        reduced_volume: The reduced volume of the mesh.
    Returns:
        mesh: a trimesh object containing the triangle mesh. The scale of the mesh is arbitrary.
    """
    fname = os.path.join(DATA_PATH, f"S0_v_{reduced_volume}_L_{subdivisions}.off")

    if not os.path.isfile(fname):
        raise ValueError(f"Could not fine a mesh with subdivisions {subdivisions}.")

    return _load_mesh(fname)


def load_equilibrium_mesh(subdivisions: int=4,
                          reduced_volume: float=DEFAULT_STRESS_FREE_REDUCED_VOLUME):
    """
    Load the default equilibrium mesh of a healthy RBC.

    Arguments:
        subdivisions: The refinement of the mesh.
        reduced_volume: The reduced volume of the corresponding stress-free mesh.
    Returns:
        mesh: a trimesh object containing the triangle mesh. The scale of the mesh is arbitrary.
    """
    fname = os.path.join(DATA_PATH, f"eq_v_{reduced_volume}_L_{subdivisions}.off")

    if not os.path.isfile(fname):
        raise ValueError(f"Could not fine a mesh with subdivisions {subdivisions}.")

    return _load_mesh(fname)
