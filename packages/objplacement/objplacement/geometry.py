#!/usr/bin/env python

import numpy as np


class DomainGeometry:
    """
    Periodic box [0, Lx] X [0, Ly] X [0, Lz] .
    Parent class of all the other geometries.
    """
    def __init__(self, L):
        """
        Arguments:
            L: Domain size
        """
        self.L = np.array(L)
        if len(self.L) != 3 or np.min(self.L) <= 0:
            raise ValueError(f"domain dimensions must have 3 non negative components.")

    def points_inside(self, positions: np.ndarray) -> np.ndarray:
        if len(positions.shape) != 2 or positions.shape[1] != 3:
            raise ValueError("Wrong dimensions: expected shape (*,3), got {positions.shape}")


        inx = np.logical_and(positions[:,0] > 0, positions[:,0] < self.L[0])
        iny = np.logical_and(positions[:,1] > 0, positions[:,1] < self.L[1])
        inz = np.logical_and(positions[:,2] > 0, positions[:,2] < self.L[2])
        return np.logical_and(inx, np.logical_and(iny, inz))

    def spheres_inside(self, positions: np.ndarray, radius: float) -> np.ndarray:
        return self.points_inside(positions)

    def volume(self):
        return np.product(self.L)


class Cylinder(DomainGeometry):
    """
    A periodic cylinder oriented along x, y or z.
    """

    def __init__(self, L,
                 radius: float,
                 center, axis: int):
        """
        L: domain size
        radius: Radius of the cylinder
        center: A point in 3D that passes through the principal axis of the cylinder.
        axis: The orientation of the cylinder (0, 1 or 2).
        """
        super().__init__(L)
        self.radius = radius
        self.center = np.array(center)
        self.axis = axis

        if len(self.center) != 3:
            raise ValueError(f"Cylinder: center must be in R^3 (got {center})")

        if not (axis == 0 or axis == 1 or axis == 2):
            raise ValueError(f"Cylinder: axis must be in [0, 1, 2] (got {axis})")

        for i in range(3):
            if i == axis:
                continue
            if self.center[i] + radius > self.L[i] or self.center[i] - radius < 0:
                raise ValueError("The cylinder does not fit in the domain.")



    def points_inside(self, positions: np.ndarray) -> np.ndarray:
        """
        State which positions are inside the cylinder.
        Arguments:
            positions: an array of positions in 3D.
        Return:
            An array of booleans, ith entry is True if ith position is inside and False otherwise.
        """
        pre = super().points_inside(positions)
        x = positions - self.center
        x[:,self.axis] = 0
        r2 = np.sum(x**2, axis=1)
        return np.logical_and(pre, r2 < self.radius**2)

    def spheres_inside(self, positions: np.ndarray, radius: float) -> np.ndarray:
        pre = super().spheres_inside(positions, radius)
        x = positions - self.center
        x[:,self.axis] = 0
        r2 = np.sum(x**2, axis=1)
        return np.logical_and(pre, r2 < (self.radius-radius)**2)


    def volume(self):
        return self.L[self.axis] * np.pi * self.radius**2


class Box(DomainGeometry):
    """
    4 plates.
    Can be outside the domain, to ease the description of parallel plates or ducts..
    """

    def __init__(self, L, lo, hi):
        """
        L: domain size
        lo: Lower corner of the box
        hi: Higher corner of the box
        """
        super().__init__(L)
        self.lo = np.array(lo)
        self.hi = np.array(hi)

        if len(self.lo) != 3 or len(self.hi) != 3:
            raise ValueError(f"Box: lo and hi must be in R^3 (got {lo}, {hi})")

        for l, h in zip(self.lo, self.hi):
            if l >= h:
                raise ValueError(f"Box: lo must be smaller than hi (got {lo}, {hi})")


    def points_inside(self, positions: np.ndarray) -> np.ndarray:
        """
        State which positions are inside the domain.
        Arguments:
            positions: an array of positions in 3D.
        Return:
            An array of booleans, ith entry is True if ith position is inside and False otherwise.
        """
        is_inside = super().points_inside(positions)
        for i in range(3):
            in_i = np.logical_and(positions[:,i] >= self.lo[i], positions[:,i] < self.hi[i])
            is_inside = np.logical_and(in_i, is_inside)
        return is_inside

    def spheres_inside(self, positions: np.ndarray, radius: float) -> np.ndarray:
        is_inside = super().spheres_inside(positions, radius)
        for i in range(3):
            in_i = np.logical_and(positions[:,i] >= self.lo[i] + radius, positions[:,i] < self.hi[i] - radius)
            is_inside = np.logical_and(in_i, is_inside)
        return is_inside


    def volume(self):
        lo = np.array([ max([self.lo[i], 0]) for i in range(3) ])
        hi = np.array([ min([self.hi[i], self.L[i]]) for i in range(3) ])
        return np.product(hi-lo)


class GeometryWithHoles(DomainGeometry):
    """
    A geometry with rectangular holes at specific positions.
    """

    def __init__(self,
                 base: DomainGeometry,
                 hole_positions: np.ndarray,
                 hole_extents: np.ndarray):
        super().__init__(base.L)
        self.base = base
        self.hole_positions = hole_positions
        self.hole_extents = hole_extents

    def points_inside(self, positions: np.ndarray) -> np.ndarray:
        is_inside = self.base.points_inside(positions)

        for r in self.hole_positions:
            lo = r - self.hole_extents/2
            hi = r + self.hole_extents/2

            outside_hole = np.full_like(is_inside, False)

            for i in range(3):
                outside_i = np.logical_or(positions[:,i] < lo[i],
                                          positions[:,i] >= hi[i])
                outside_hole = np.logical_or(outside_hole, outside_i)

            is_inside = np.logical_and(is_inside, outside_hole)

        return is_inside

    def spheres_inside(self, positions: np.ndarray, radius: float) -> np.ndarray:
        """
        For simplicity, only the bounding box of the sphere is checked against the holes.
        """

        is_inside = self.base.spheres_inside(positions, radius)

        for r in self.hole_positions:
            lo = r - self.hole_extents/2
            hi = r + self.hole_extents/2

            outside_hole = np.full_like(is_inside, False)

            for i in range(3):
                # TODO this is not correct but simpler
                outside_i = np.logical_or(positions[:,i] < lo[i] - radius,
                                          positions[:,i] >= hi[i] + radius)
                outside_hole = np.logical_or(outside_hole, outside_i)

            is_inside = np.logical_and(is_inside, outside_hole)

        return is_inside


    def volume(self):
        return self.base.volume()


class FromSdfTools(DomainGeometry):
    """
    Shape described by a signed distance function field (SDF).
    The inside is negative and the outside is positive.
    This module assumes the interface of the sdf_tools package:
    https://sdftools.readthedocs.io/en/latest/
    """
    def __init__(self, L, sdf):
        """
        Arguments:
            L: Domain size
            sdf: SDF field function.
        """
        super().__init__(L)
        self.sdf = sdf

    def points_inside(self, positions: np.ndarray) -> np.ndarray:
        is_inside = super().points_inside(positions)
        for i, pos in enumerate(positions):
            s = self.sdf.at(tuple(pos))
            is_inside[i] = is_inside[i] and s < 0
        return is_inside

    def spheres_inside(self, positions: np.ndarray, radius: float) -> np.ndarray:
        is_inside = super().spheres_inside(positions, radius)
        for i, pos in enumerate(positions):
            s = self.sdf.at(tuple(pos))
            is_inside[i] = is_inside[i] and s < -radius
        return is_inside

    def volume(self):
        V0 = super().volume()
        n = 100000
        positions = np.random.uniform(low=(0, 0, 0), high=self.L, size=(n,3))
        num_inside = 0
        for pos in positions:
            s = self.sdf.at(tuple(pos))
            if s < 0:
                num_inside += 1
        return V0 * num_inside / n
