from typing import Callable, List, Tuple

import numpy as np

from src.airfoil import Airfoil
from src.types import Panel, PointValue


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
    def construct_coeff_matrix(
        cls,
        panels: List[Panel],
        airfoil: Airfoil,
        alpha: float,
    ) -> np.ndarray:
        """Construct coefficient (LHS) A matrix.

        Args:
            panels:
                Panels representing the rotated airfoil geometry.
            airfoil:
                Original airfoil object.
            alpha:
                Airfoil angle of attack in deg.

        Returns:
            Coefficient matrix with shape (N + 1, N + 1), where N is the number of panels.
        """
        num_panels = len(panels)

        if num_panels == 0:
            raise ValueError("At least one panel is required")

        latest_shed_vortex_pos = cls.get_latest_shed_vortex_position(
            airfoil,
            alpha,
        )

        # Prealloc (N+1) x (N+1) matrix
        coeff_matrix = np.empty(
            (num_panels + 1, num_panels + 1),
            dtype=float,
        )

        for i, control_panel in enumerate(panels):
            control_point = control_panel.control_point_position
            normal = control_panel.normal

            # Influence of every bound vortex on control point i.
            for j, vortex_panel in enumerate(panels):
                induced_velocity = cls.vel_induced_by_vortex(
                    control_point,
                    vortex_panel.vortex_position,
                    1.0,
                )

                coeff_matrix[i, j] = np.dot(
                    induced_velocity,
                    normal,
                )

            # Influence of the newly shed wake vortex on control point i.
            shed_vortex_velocity = cls.vel_induced_by_vortex(
                control_point,
                latest_shed_vortex_pos,
                1.0,
            )

            coeff_matrix[i, num_panels] = np.dot(
                shed_vortex_velocity,
                normal,
            )

        # Kelvin condition
        coeff_matrix[num_panels, :] = 1.0

        return coeff_matrix

    @classmethod
    def construct_RHS_vector(
        cls,
        Q_inf: Tuple[float, Callable[[float], float]],
        panels: List[Panel],
        wake_vortices: List[PointValue],
        gamma_sum_prev: float,
        k: int,
        delta_t: float,
    ) -> np.ndarray:
        """Constructs the RHS (b) vector

        Args:
            Q_inf (float, Callable[[float], float]]): A vector representing time-varying functions of the x and z components of the freestream
            panels (List[Panel]): List of airfoil post-rotation panels
            wake_vortices (List[PointValue]): List of wake vortices from last time step, however stepped forward in positions by one time step (to make room for new latest shed vortex to be solved for). That is, all wake vortices from t(k-1) post-shifted
            gamma_sum_prev (float): the total sum of bound circulation in the previous time step
            k (int): time step
            delta_t (float): time resolution

        Returns:
            np.ndarray: the RHS (b) vector
        """
        num_panels = len(panels)

        if num_panels == 0:
            raise ValueError("At least one panel is required")

        rhs_vector = np.empty(
            (num_panels + 1,),
            dtype=float,
        )

        # time varying freestream functions, U(t) and W(t)
        # Take the time resolution
        U_inf = Q_inf[0]
        Wfunc = Q_inf[1]
        t = delta_t * (k - 1)

        for i, panel in enumerate(panels):
            control_point = panel.control_point_position
            control_point_x = control_point[0]

            # The horizontal freestream U_inf(t) is constant for a given time step
            # across the whole field domain.
            # However, the vertical gust propagates down the airfoil, and the encountered
            # local gust can be evaluated using W(t - t_n), where t_n is the time it takes
            # to reach control point n
            # Justification for this formula is included in theory
            t_n = control_point_x / U_inf
            non_induced_velocity = np.array([U_inf, Wfunc(t - t_n)])

            # Calculate wake vortex contribution
            wake_vortex_contribution = np.zeros((2,), dtype=float)
            for wake_vortex in wake_vortices:
                gamma = wake_vortex.value
                vortex_pos = wake_vortex.point

                wake_vortex_contribution = np.add(
                    wake_vortex_contribution,
                    cls.vel_induced_by_vortex(control_point, vortex_pos, gamma),
                )

            # This is Q_i
            total_field_contribution = np.add(
                non_induced_velocity, wake_vortex_contribution
            )

            rhs_vector[i] = -np.dot(total_field_contribution, panel.normal)

        # Apply Kelvin's theorem for last entry
        rhs_vector[num_panels] = gamma_sum_prev

        return rhs_vector

    @classmethod
    def develop_single_wake_vortex(
        cls,
        Q_inf: Tuple[float, Callable[[float], float]],
        current_wake: List[PointValue],
        vortex_index: int,
        bound_gamma: List[PointValue],
        delta_t: float,
        k: int,
    ):
        """Develops a single wake vortex, returning a PointValue of its new position

        Args:
            Q_inf (Tuple[float, Callable[[float], float]]): Freestream vector
            current_wake (List[PointValue]): Full solved wake at time step k
            vortex_index (int): The index of the vortex to be updated
            bound_gamma (List[PointValue]): Bound vortex distribution
            delta_t: time resolution (length of one time step)
            k: time step
        """
        if vortex_index not in range(len(current_wake)):
            raise ValueError(
                f"Supplied wake vortex index {vortex_index} is not within bounds for wake of length {len(current_wake)}"
            )

        current_vortex = current_wake[vortex_index]

        # Bound vortex velocity contribution
        bound_vortex_contribution = np.zeros((2,), dtype=float)
        for _, bound_vortex in enumerate(bound_gamma):
            contribution = cls.vel_induced_by_vortex(
                current_vortex.point, bound_vortex.point, bound_vortex.value
            )
            bound_vortex_contribution = np.add(bound_vortex_contribution, contribution)

        # Wake vortex contribution
        wake_vortex_contribution = np.zeros((2,), dtype=float)
        for i, wake_vortex in enumerate(current_wake):
            if i == vortex_index:
                continue

            contribution = cls.vel_induced_by_vortex(
                current_vortex.point, wake_vortex.point, wake_vortex.value
            )
            wake_vortex_contribution = np.add(wake_vortex_contribution, contribution)

        # freestream
        current_vortex_x = current_vortex.point[0]
        U_inf = Q_inf[0]
        Wfunc = Q_inf[1]

        t = delta_t * (k - 1)
        t_n = current_vortex_x / U_inf

        freestream_contribution = np.array([U_inf, Wfunc(t - t_n)])

        # Combine and develop singular vortex
        total_induced_velocity = (
            bound_vortex_contribution
            + wake_vortex_contribution
            + freestream_contribution
        )

        delta_pos = total_induced_velocity * delta_t
        developed_vortex_position = current_vortex.point + delta_pos

        return PointValue(developed_vortex_position, current_vortex.value)

    @classmethod
    def develop_wake(
        cls,
        Q_inf: Tuple[float, Callable[[float], float]],
        current_wake: List[PointValue],
        bound_gamma: List[PointValue],
        delta_t: float,
        k: int,
    ) -> List[PointValue]:
        """Takes in the current full wake (including latest shed vortex), and updates it according
        to the solved bound circulation distribution, and returns the new wake

        Args:
            Q_inf (Tuple[float, Callable[[float], float]]): Freestream vector
            current_wake (List[PointValue]): Full solved wake at time step k
            bound_gamma (List[PointValue]): Bound vortex distribution
            delta_t: time resolution (length of one time step)
            k: time step
        """
        new_wake = []
        for i, wake_vortex in enumerate(current_wake):
            new_wake.append(
                cls.develop_single_wake_vortex(
                    Q_inf, current_wake, i, bound_gamma, delta_t, k
                )
            )

        return new_wake
