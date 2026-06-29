import numpy as np

from src.airfoil import Airfoil


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
        """Calculates the velocity induced at p_i by a vortex located at p_v. Synonymous with Katz & Plotkin (2001) VOR2D method.

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

    @classmethod
    def rotate_point(cls, p: np.ndarray, alpha: float) -> np.ndarray:
        """Takes a point in the x-z plane and rotates it by `alpha` degrees clockwise about the origin.
        This is to 'tilt' a point with a specific angle of attack (alpha)

        Args:
            p (np.ndarray): the point [x, z] to transform
            alpha (float): Angle of attack (deg)

        Returns:
            np.ndarray: The newly rotated point [x', z']
        """
        alpha_rad = np.deg2rad(alpha)
        rotation_matrix = np.array(
            [
                [np.cos(alpha_rad), np.sin(alpha_rad)],
                [-np.sin(alpha_rad), np.cos(alpha_rad)],
            ]
        )
        return rotation_matrix @ p

    @classmethod
    def rotate_points(cls, points: np.ndarray, alpha: float) -> np.ndarray:
        """Takes in an array of points and rotates them clockwise by `alpha` degrees about the origin.

        Args:
            points (np.ndarray): (N, 2) size array (array of points, which are size (2,) `np.ndarray`'s)
            alpha (float): Angle of attack (deg)

        Returns:
            np.ndarray: (N, 2) size array (array of transformed points, which are size (2,) `np.ndarray`'s)
        """
        return np.array([cls.rotate_point(p, alpha) for p in points])

    @classmethod
    def get_latest_shed_vortex_position(
        cls, airfoil: Airfoil, alpha: float
    ) -> np.ndarray:
        """Gets the position for the latest shed vortex Gamma_w_k
        Using findings from Bernard T. Roesler and Brenden P. Epps, places this vortex at 1/4 panel length aft of the TE

        Args:
            airfoil (Airfoil): The airfoil itself
            alpha (float): The angle of attack (deg)

        Returns:
            float: the position (in m, from the LE) where the latest shed vortex is, post-AoA transform
        """
        # Ignoring AoA, we place the vortex 1/4 panel length aft of the chord (which is along the x-axis)
        pre_transform_x = airfoil.c + 1 / 4 * airfoil.delta_x
        pre_transform_z = 0
        vortex_pos = np.array([pre_transform_x, pre_transform_z])

        return cls.rotate_point(vortex_pos, alpha)

    @classmethod
    def construct_coeff_matrix(cls, airfoil: Airfoil):
        pass
