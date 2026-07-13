from typing import List, Optional, Callable, Tuple
import numpy as np

from src.airfoil import Airfoil
from src.vortex_strategy import VortexLumpingStrategy
from src.types import Panel, PointValue
from src.util import SolverUtils


class SolveResult:
    """
    A simple result container returned by Solver.solve(...).
    """

    def __init__(
        self,
        bound_gamma_history: List[List[PointValue]],
        wake_gamma_history: List[List[PointValue]],
        lift_history: Optional[List[float]] = None,
    ):
        """
        :param bound_gamma_history: List of k rows containing bound vortex distribution over time.
                                    Each row is a vector of gamma values associated with points.
        :param wake_gamma_history: List of k rows containing wake vortex distribution over time.
                                   Each row is a vector of gamma values associated with points.
        :param lift_history: List of lift evaluations across time steps.
        """
        self.bound_gamma_history = bound_gamma_history
        self.wake_gamma_history = wake_gamma_history
        self.lift_history = lift_history


class Solver:
    """
    Represents a single solver scenario. Stores all information needed to solve.
    """

    def __init__(
        self,
        airfoil: Airfoil,
        strategy: Optional[VortexLumpingStrategy],
        num_time_steps: int,
        Q_inf: Tuple[float, Callable[[float], float]],
        alpha: float,
    ):
        """
        :param airfoil: The airfoil being simulated.
        :param strategy: The vortex lumping strategy to use, or None to disable lumping.
        :param num_time_steps: Number of time steps to solve over.
        :param Q_inf: A 2D vector containing [U_inf (scalar), W(t)], defining time-variable vertical gust with a constant freestream
        :param alpha: the angle-of-attack (deg) to be used in this simulation (note: AoA is constant)
        """
        self.airfoil = airfoil
        self.strategy = strategy
        self.num_time_steps = num_time_steps
        self.Q_inf = Q_inf
        self.alpha = alpha

        self.delta_t = np.nan
        self.airfoil_panels: List[Panel] = []

        self.bound_gamma_history: List[List[PointValue]] = []
        self.bound_gamma_sum_history: List[float] = []
        self.wake_gamma_history: List[List[PointValue]] = []
        self.lift_history: List[float] = []

    def solve(self) -> SolveResult:
        """
        Solves the configured scenario.

        :param get_lift_history: Whether to compute lift history alongside circulation.
        :return: SolveResult containing the computed histories.
        """
        self.clear()
        panels = self.preprocess_panels()  # panels are post-rotation
        _ = self.preprocess_Qinf()

        # The LHS (A) coefficient matrix is actually time-invariant, as geometry of airfoil
        # stays constant. Thus this only needs to be evaluated once at the beginnig
        Amat = SolverUtils.construct_coeff_matrix(panels, self.airfoil, self.alpha)

        # Every time step yields a solution to the wake, including the just-solved-for latest shed vortex.
        # In order for the next time step to be solved, the wake is developed, 'shifting' all the wake vortices
        # from time step k.
        # That is, to solve time step k + 1, the wake from time step k must be developed first.
        # However, self.wake_gamma_history stores the wake, pre-developed, at time step k
        # This variable tracks the post-developed wake from the previous time step.
        current_wake: List[PointValue] = []

        # Main loop
        for k in range(1, self.num_time_steps + 1):
            current_wake = self.solve_singular_timestep(k, Amat, current_wake)

        return SolveResult(
            bound_gamma_history=self.bound_gamma_history,
            wake_gamma_history=self.wake_gamma_history,
            lift_history=self.lift_history,
        )

    # Helper functions for Solver class
    def clear(self):
        """Clears the histories for fresh use"""
        self.bound_gamma_history = []
        self.bound_gamma_sum_history = []
        self.wake_gamma_history = []
        self.lift_history = []

    def preprocess_panels(self) -> List[Panel]:
        """
        Preprocesses simulation parameters, ie. rotating the airfoil and converting the camber
        line points into a series of panels (post-rotation) for use.
        Returns these panels.
        """
        rotated_camber_points = SolverUtils.rotate_points(
            self.airfoil.camber(), self.alpha
        )

        panels = []
        for i in range(len(rotated_camber_points) - 1):
            p1 = rotated_camber_points[i]
            p2 = rotated_camber_points[i + 1]

            panels.append(Panel(p1, p2))

        # set self attribute in case we need it later
        self.airfoil_panels = panels
        return panels

    def preprocess_Qinf(self) -> float:
        """Preprocesses the given Q_inf function vector. Check that U_inf is positive.
        Note: If surging flow is to be added, calculating the relevant W(t) position will involve
        integrating U(t)

        Returns:
            float: delta_t - the time resolution
        """
        U_inf = self.Q_inf[0]
        if U_inf <= 0:
            raise ValueError(
                "Supplied Q_inf U(t) component evaluates to 0 at t=0. Please specify a horizantal freestream function which stays strictly positive over the time domain."
            )

        self.delta_t = self.airfoil.delta_x / U_inf
        return self.delta_t

    def solve_singular_timestep(
        self, k: int, Amat: np.ndarray, current_wake: List[PointValue]
    ) -> List[PointValue]:
        """Solves bound circulation and latest shed vortex for time step k, and returns the
        wake of timestep k shifted for use in time step k + 1.
        Also updates the solver class's history arrays

        Args:
            k (int): time step
            Amat (np.ndarray): The coefficient (A) matrix
            current_wake (List[PointValue]): The state of the wake at time step k, minus the latest shed vortex.
                This is the full wake of time step k - 1, but developed

        Raises:
            RuntimeError: If the number of solved bound vortices does not equal the number of panels (safety check)

        Returns:
            List[PointValue]: The developed wake of time step k, to be used in time step k + 1
        """
        # Base case Kelvin condition
        if k == 1:
            prev_gamma_sum = 0
        else:
            prev_gamma_sum = self.bound_gamma_sum_history[k - 2]  # k is 1-indexed

        rhs_vec = SolverUtils.construct_RHS_vector(
            self.Q_inf,
            self.airfoil_panels,
            current_wake,
            prev_gamma_sum,
            k,
            self.delta_t,
        )

        # solve Amat \ rhs_vec
        solution_k = np.linalg.solve(Amat, rhs_vec)
        bound_gamma_sol = solution_k[:-1].copy()

        if len(bound_gamma_sol) != len(self.airfoil_panels):
            raise RuntimeError(
                f"Length mismatch of bound gamma with length {len(bound_gamma_sol)} and number of panels {len(self.airfoil_panels)}"
            )

        latest_shed_vortex_gamma = solution_k[-1]
        latest_shed_vortex_pos = SolverUtils.get_latest_shed_vortex_position(
            self.airfoil, self.alpha
        )
        latest_shed_vortex = PointValue(
            latest_shed_vortex_pos, latest_shed_vortex_gamma
        )

        # Wake solution at time step k
        wake_sol_k = [latest_shed_vortex, *current_wake]

        # Update self histories
        # Gamma sum history
        gamma_sum = sum(bound_gamma_sol)
        self.bound_gamma_sum_history.append(gamma_sum)

        # Bound gamma history
        bound_gamma_points_sol = []
        for i, gamma_i in enumerate(bound_gamma_sol):
            bound_vortex_pos = self.airfoil_panels[i].vortex_position
            bound_gamma_points_sol.append(PointValue(bound_vortex_pos, gamma_i))
        self.bound_gamma_history.append(bound_gamma_points_sol)

        # Wake gamma history
        self.wake_gamma_history.append(wake_sol_k)

        # Return developed wake
        return SolverUtils.develop_wake(
            self.Q_inf, wake_sol_k, bound_gamma_points_sol, self.delta_t, k
        )
