#!/usr/bin/env python

import numpy as np
from .geometry import DomainGeometry

def generate_random_quaternions(n: int, seed: int=23425334):
    """
    Generate random orientations
    See https://stackoverflow.com/a/44031492/11630848

    Arguments:
        n: Number of quaternions to generate.
        seed: random seed
    Return:
        A np.nd array of shape (n,4) filled with random quaternions
    """

    np.random.seed(seed)
    u = np.random.uniform(0, 1, n)
    v = np.random.uniform(0, 1, n)
    w = np.random.uniform(0, 1, n)

    q = np.zeros((n, 4))

    q[:,0] = np.sqrt(1-u) * np.sin(2 * np.pi * v)
    q[:,1] = np.sqrt(1-u) * np.cos(2 * np.pi * v)
    q[:,2] = np.sqrt(u) * np.sin(2 * np.pi * w)
    q[:,3] = np.sqrt(u) * np.cos(2 * np.pi * w)

    return q


def generate_positions(*,
                       geometry: "DomainGeometry",
                       obj_volume: float,
                       obj_extents: np.ndarray,
                       target_volume_fraction: float,
                       seed: int=23425334,
                       safety: float=1.1):
    """
    Try to place objects to achieve a given volume fraction.
    """
    obj_extents = np.array(obj_extents)

    # number of objects required
    V = geometry.volume()
    nrequired = int (round(target_volume_fraction * V / obj_volume))

    L = geometry.L
    n = np.floor(L / (obj_extents * safety)).astype(int)
    h = L / n

    x = np.linspace(h[0]/2, L[0]-h[0]/2, n[0], endpoint=True)
    y = np.linspace(h[1]/2, L[1]-h[1]/2, n[1], endpoint=True)
    z = np.linspace(h[2]/2, L[2]-h[2]/2, n[2], endpoint=True)
    x, y, z = np.meshgrid(x, y, z, indexing='ij')

    x = x.flatten()
    y = y.flatten()
    z = z.flatten()
    pos = np.zeros((len(x), 3))
    pos[:,0] = x
    pos[:,1] = y
    pos[:,2] = z

    radius = 0.5 * max(obj_extents)
    idx = np.argwhere(geometry.spheres_inside(pos, radius)).flatten()

    if nrequired > len(idx):
        raise RuntimeError(f"could not generate positions: can generate only {len(idx)}, required {nrequired}")

    np.random.seed(seed)
    np.random.shuffle(idx)

    return pos[idx[:nrequired],:]


def generate_ic(*,
                geometry: "DomainGeometry",
                obj_volume: float,
                obj_extents: np.ndarray,
                target_volume_fraction: float,
                orientation=None,
                seed: int=23425334,
                safety: float=1.1):
    """
    Generate initial conditions (positions and orientations) of non overlapping objects in the given geometry.

    Arguments:
        geometry: the domain where the objects can be placed.
        obj_volume: volume of one object to be placed.
        obj_extents: Extents of one object.
        target_volume_fraction: The desired volume fraction of the objects in the given geometry.
        orientation: if set, the quaternion that describes the initial rotation of all objects. else, will be a random quaternion.
        seed: The random seed to choose the sites and orientation.
        safety: controls the spacing between objects. Must be larger than 1 to guarantee no overlap.

    Returns:
        com_q: an array of positions and quaternions.

    Note:
        * If orientation is not set (or set to None), the extents of the objects will be considered as larger to guarantee no overlap
          due to the random orientation. If a higher volume fraction is required, setting the orientation can help.
        * If orientation is set, the object extents must be consistent with the oriented object. Otherwise overlap might occur between objects.
        * if this function does not manage to place enough objects to achieve the desired volume fraction, an exception will be thrown.
    """

    if orientation is None:
        obj_extents = np.full_like(obj_extents, np.max(obj_extents))

    pos = generate_positions(geometry=geometry,
                             obj_volume=obj_volume,
                             obj_extents=obj_extents,
                             target_volume_fraction=target_volume_fraction,
                             seed=seed,
                             safety=safety)

    if orientation is None:
        quat = generate_random_quaternions(n=len(pos), seed=seed)
    else:
        q = list(orientation)
        if len(q) != 4:
            raise ValueError(f"if set, orientation must be a quaternion (length 4); got {orientation}")
        quat = len(pos) * [q]



    return [[r[0], r[1], r[2], q[0], q[1], q[2], q[3]] for r, q in zip(pos, quat)]
