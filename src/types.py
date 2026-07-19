from dataclasses import dataclass

import numpy as np


class Panel:
    """
    Class representing a panel along an airfoil camber line.
    Contains information on the points defining the panel, the positions of the bound vortex
    and control points, as well as unit vectors orthonormal / tangent to the panel.
    """

    def __init__(self, p1: np.ndarray, p2: np.ndarray):
        self.p1 = np.asarray(p1, dtype=float).copy()
        self.p2 = np.asarray(p2, dtype=float).copy()

        if self.p1.shape != (2,) or self.p2.shape != (2,):
            raise ValueError("p1 and p2 must each be 2D points with shape (2,)")

        self.vector = self.p2 - self.p1
        length = np.linalg.norm(self.vector)

        self.length = length

        if np.isclose(length, 0.0):
            raise ValueError("A panel cannot have zero length")

        # Unit vector directed from p1 to p2.
        self.tangent = self.vector / length

        # Left-hand unit normal relative to the p1 -> p2 direction.
        self.normal = np.array([-self.tangent[1], self.tangent[0]])

        # Quarter-panel vortex and three-quarter-panel control point.
        self.vortex_position = self.p1 + 0.25 * self.vector
        self.control_point_position = self.p1 + 0.75 * self.vector

        # Angle the panel makes with the positive x-axis
        # Positive angles are counterclockwise
        self.alpha_rad = float(np.arctan2(self.vector[1], self.vector[0]))
        self.alpha_deg = np.rad2deg(self.alpha_rad)


@dataclass(slots=True, eq=False)
class PointValue:
    """A scalar value associated with a point in the 2D x-z plane."""

    point: np.ndarray
    value: float

    def __post_init__(self) -> None:
        self.point = np.asarray(self.point, dtype=float).copy()

        if self.point.shape != (2,):
            raise ValueError("point must be a 2D point with shape (2,)")

        self.value = float(self.value)

    @property
    def x(self) -> float:
        return float(self.point[0])

    @property
    def z(self) -> float:
        return float(self.point[1])
