#!/usr/bin/env python

import math

def average_intervertex_distance(*,
                                 area: float,
                                 nv: int) -> float:
    """
    Compute the average distance between 2 vertices, l0, on a triangle mesh of given area.
    A / nt = sqrt(3) l0^2 / 4
    Parameters:
        area: Area of the surface
        nv: Number of vertices
    Returns:
        The average edge length.
    """
    nt = 2 * nv - 4
    l0 = (4 * area / (3**0.5 * nt))**0.5
    return l0


def equivalent_sphere_radius(*,
                             area: float) -> float:
    """
    Compute the radius of the sphere with the given area.
    Parameters:
        area: The area of the surface.
    Returns:
        RA: the radius that the sphere of given area would have.
    """
    RA = (area / (4*math.pi))**(1/2)
    return RA


def reduced_volume(*,
                   area: float,
                   volume: float) -> float:
    """
    Compute the ratio of the volume of a cell and that of the equivalent sphere with same area.
    Parameters:
        area: The area of the cell.
        volume: The volume of the cell.
    Returns:
        v: the radius that the sphere of given area would have.
    """
    RA = equivalent_sphere_radius(area=area)
    VA = 4 * math.pi / 3 * RA**3
    v = VA / volume
    return v
