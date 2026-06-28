import numpy as np


class SolverUtils:
    """
    Utility class containing universal solver helper methods.
    """

    @classmethod
    def unit_vector_normal_to(cls, v: np.ndarray) -> np.ndarray:
        """
        Returns a unit vector normal to a given 2D vector.

        :param v: A 2-dimensional vector.
        :return: A 2-dimensional normalized normal vector.
        """
        v = np.asarray(v, dtype=float)
        if v.shape != (2,):
            raise ValueError("Input vector must be 2-dimensional.")

        normal = np.array([-v[1], v[0]])
        norm = np.linalg.norm(normal)
        if norm == 0:
            return normal
        return normal / norm

    @classmethod
    def vel_induced_by_vortex(
        cls, p_i: np.ndarray, p_v: np.ndarray, gamma: float
    ) -> np.ndarray:
        """Calculates the velocity induced at p_i by a vortex located at p_v

        Args:
            p_i (np.ndarray): Location of point where the induced velocity is calculated
            p_v (np.ndarray): Location of the vortex
            gamma (float): Strength of the vortex

        Returns:
            np.ndarray: (2,) vector representing (u,w), ie. the velocity induced.
        """

        x_i, z_i = p_i
        x_v, z_v = p_v

        r_squared = (x_i - x_v) ** 2 + (z_i - z_v) ** 2

        rhs_mat = np.array([[0, 1], [-1, 0]])
        rhs_vec = np.array(
            [
                x_i - x_v,
                z_i - z_v,
            ]
        )

        return gamma / (2 * np.pi * r_squared) * (rhs_mat @ rhs_vec)
