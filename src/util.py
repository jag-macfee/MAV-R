from typing import Callable, List, Optional, Sequence, Tuple

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
    def calculate_Q_i(
        cls,
        Q_inf: Tuple[float, Callable[[float], float]],
        wake_vortices: List[PointValue],
        panel: Panel,
        t: float,
    ) -> np.ndarray:
        """Calculates Q_i, the total field contribution on a given panel's control point.
        This includes velocity contribution from the freestream, and gust, as well as wake vortex induced velocity.

        Args:
            Q_inf (Tuple[float, Callable[[float], float]]): Freestream vector
            wake_vortices (List[PointValue]): Wake vortices relevant for calculation
            panel (Panel): The panel to be calculated at
            t (float): time since first time step

        Returns:
            np.ndarray: The field contribution to velocity at the specified panel's control point
        """
        U_inf = Q_inf[0]
        Wfunc = Q_inf[1]

        control_point = panel.control_point_position
        control_point_x = control_point[0]

        # The horizontal freestream U_inf is constant for a given time step
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
        return total_field_contribution

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

        # Time passed since first step
        t = delta_t * (k - 1)

        for i, panel in enumerate(panels):
            total_field_contribution = cls.calculate_Q_i(Q_inf, wake_vortices, panel, t)

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


class ResultUtils:
    """Utilities for post-processing a completed solver result."""

    @classmethod
    def extract_lift_frequency_response(
        cls,
        lift_history: Sequence[float],
        delta_t: float,
        U_inf: float,
        chord: float,
        rho: float,
        v_0: float,
        max_normalised_frequency: Optional[float] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        r"""Extract the Lysak-normalised lift frequency response.

        The supplied lift history must come from the existing discrete impulse
        gust calculation. No modification is made to the impulse definition.

        Following Lysak's Eqs. (2.51)--(2.52), the discrete Fourier transform is

        ``L_hat(f) = (delta_t / (2*pi)) * DFT[L_k]``.

        For an input gust ``v_0 * delta(k - 1)``, its transform is
        ``v_0 * delta_t / (2*pi)``. Dividing the output transform by the input
        transform therefore gives the lift/gust transfer function. The returned
        ordinate is the paper's dimensionless squared magnitude,

        ``|H(f)|^2 / (pi * rho * c * U_inf)^2``.

        This is algebraically equivalent to

        ``|DFT[L_k]|^2 / (pi * rho * c * v_0 * U_inf)^2``.

        The frequency coordinate is normalised as ``f*c/U_inf``. The zero-
        frequency term is omitted because the intended plot uses logarithmic
        axes.

        Args:
            lift_history:
                Total lift per unit span at each solver time step.
            delta_t:
                Solver time-step length in seconds.
            U_inf:
                Positive horizontal freestream velocity.
            chord:
                Positive airfoil chord length.
            rho:
                Positive fluid density.
            v_0:
                Non-zero amplitude multiplying the existing discrete impulse.
            max_normalised_frequency:
                Optional upper limit on ``f*c/U_inf``. Lysak recommends
                ``N/4`` for an ``N``-panel model, even though the Nyquist limit
                is ``N/2``.

        Returns:
            ``(normalised_frequency, normalised_response_squared)``.

        Raises:
            ValueError:
                If the lift history or physical/scaling values are invalid, or
                if no positive finite frequencies remain after filtering.
        """
        lift = np.asarray(lift_history, dtype=float)

        if lift.ndim != 1:
            raise ValueError("lift_history must be a one-dimensional sequence")
        if lift.size < 2:
            raise ValueError("lift_history must contain at least two samples")
        if np.any(~np.isfinite(lift)):
            raise ValueError("lift_history must contain only finite values")

        delta_t = float(delta_t)
        U_inf = float(U_inf)
        chord = float(chord)
        rho = float(rho)
        v_0 = float(v_0)

        if not np.isfinite(delta_t) or delta_t <= 0.0:
            raise ValueError("delta_t must be finite and strictly positive")
        if not np.isfinite(U_inf) or U_inf <= 0.0:
            raise ValueError("U_inf must be finite and strictly positive")
        if not np.isfinite(chord) or chord <= 0.0:
            raise ValueError("chord must be finite and strictly positive")
        if not np.isfinite(rho) or rho <= 0.0:
            raise ValueError("rho must be finite and strictly positive")
        if not np.isfinite(v_0) or np.isclose(v_0, 0.0):
            raise ValueError("v_0 must be finite and non-zero")

        if max_normalised_frequency is not None:
            max_normalised_frequency = float(max_normalised_frequency)
            if (
                not np.isfinite(max_normalised_frequency)
                or max_normalised_frequency <= 0.0
            ):
                raise ValueError("max_normalised_frequency must be finite and positive")

        # Lysak Eq. (2.51): Fourier transform of the lift output.
        transform_scale = delta_t / (2.0 * np.pi)
        lift_transform = transform_scale * np.fft.rfft(lift)

        gust_transform = v_0 * transform_scale
        frequency_response = lift_transform / gust_transform

        frequency_hz = np.fft.rfftfreq(lift.size, d=delta_t)
        normalised_frequency = frequency_hz * chord / U_inf

        reference_response = np.pi * rho * chord * U_inf
        normalised_response_squared = (
            np.abs(frequency_response) ** 2 / reference_response**2
        )

        valid = (
            (normalised_frequency > 0.0)
            & np.isfinite(normalised_frequency)
            & np.isfinite(normalised_response_squared)
            & (normalised_response_squared > 0.0)
        )

        if max_normalised_frequency is not None:
            valid &= normalised_frequency <= max_normalised_frequency

        normalised_frequency = normalised_frequency[valid]
        normalised_response_squared = normalised_response_squared[valid]

        if normalised_frequency.size == 0:
            raise ValueError(
                "no positive finite frequency-response values remain after filtering"
            )

        return normalised_frequency, normalised_response_squared

    @classmethod
    def sears_response_squared_approximation(
        cls,
        normalised_frequency: Sequence[float],
        m: float = 1.3,
    ) -> np.ndarray:
        r"""Evaluate Lysak's approximation to the squared Sears magnitude.

        Args:
            normalised_frequency:
                Values of ``f*c/U_inf``.
            m:
                Approximation exponent. Lysak uses ``m = 1.3``.

        Returns:
            An array containing ``|S(pi*f*c/U_inf)|^2``.
        """
        frequency = np.asarray(normalised_frequency, dtype=float)
        m = float(m)

        if frequency.ndim != 1:
            raise ValueError("normalised_frequency must be a one-dimensional sequence")
        if frequency.size == 0:
            raise ValueError("normalised_frequency must contain at least one value")
        if np.any(~np.isfinite(frequency)) or np.any(frequency < 0.0):
            raise ValueError(
                "normalised_frequency must contain finite non-negative values"
            )
        if not np.isfinite(m) or m <= 0.0:
            raise ValueError("m must be finite and strictly positive")

        sears_argument = np.pi * frequency
        return (1.0 / (1.0 + (2.0 * np.pi * sears_argument) ** m)) ** (1.0 / m)
