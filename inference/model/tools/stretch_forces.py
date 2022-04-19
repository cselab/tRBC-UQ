#! /usr/bin/env python

import numpy as np

def compute_mean_edge_dist(mesh):
    """
    Compute the average distance between neighboring vertices of the given mesh.
    """
    vertices = np.array(mesh.vertices)
    faces    = np.array(mesh.faces)

    def dist(v, ids1, ids2):
        d = np.sqrt(np.sum((v[ids1,:] - v[ids2,:])**2, axis=1))
        return np.mean(d)

    d0 = dist(vertices, faces[:,0], faces[:,1])
    d1 = dist(vertices, faces[:,0], faces[:,2])
    d2 = dist(vertices, faces[:,2], faces[:,1])
    return np.mean([d0, d1, d2])

def compute_micro_beads_forces(mesh,
                               contact_diameter: float,
                               bead_force: float):
    """
    Compute the forces exerted by beads on a cell.
    The forces are oriented along the x direction.

    Args:
        mesh: The mesh of the membrane, aligned in the xy plane (trimesh).
        contact_diameter: Diameter of the disk of contact between the bead and the membrane.
        bead_force: The magnitude of the total force exerted by each bead on the cell.
    Returns:
        forces: The array that contains the forces per vertex
    """
    vertices = np.array(mesh.vertices)
    h = compute_mean_edge_dist(mesh)

    vertices[:,0] -= np.mean(vertices[:,0])

    # RBC radius
    R0 = np.ptp(vertices[:,0]) / 2
    # contact radius
    rc = contact_diameter / 2

    # find the contact line distance from center of the cell
    xcontact = np.sqrt(R0**2 - rc**2)

    # select vertices that are within +- h/2 from this contact line
    x = vertices[:,0]
    idx_right = np.argwhere(np.abs(x - xcontact) <= h/4)
    idx_left  = np.argwhere(np.abs(x + xcontact) <= h/4)

    forces = np.zeros_like(vertices)

    forces[idx_right, 0] = +bead_force / len(idx_right)
    forces[idx_left,  0] = -bead_force / len(idx_left)

    return forces
