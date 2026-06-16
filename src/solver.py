from typing import List, Optional, Callable, Tuple
import numpy as np

from src.airfoil import Airfoil
from src.vortex_strategy import VortexLumpingStrategy


class SolveResult:
    """
    A simple result container returned by Solver.solve(...).
    """

    def __init__(
        self,
        bound_gamma_history: List[np.ndarray],
        wake_gamma_history: List[np.ndarray],
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
        Q_inf: Tuple[Callable[[float], float], Callable[[float], float]],
    ):
        """
        :param airfoil: The airfoil being simulated.
        :param strategy: The vortex lumping strategy to use, or None to disable lumping.
        :param num_time_steps: Number of time steps to solve over.
        :param Q_inf: A 2D vector containing functions of t describing freestream/gust shape.
        """
        self.airfoil = airfoil
        self.strategy = strategy
        self.num_time_steps = num_time_steps
        self.Q_inf = Q_inf

        self.bound_gamma_history: List[np.ndarray] = []
        self.wake_gamma_history: List[np.ndarray] = []
        self.lift_history: List[float] = []

    def solve(self) -> SolveResult:
        """
        Solves the configured scenario.

        :param get_lift_history: Whether to compute lift history alongside circulation.
        :return: SolveResult containing the computed histories.
        """
        self.bound_gamma_history = []
        self.wake_gamma_history = []
        self.lift_history = []

        # Stub implementation of integration over self.num_time_steps and
        # interactions with self.strategy. Each time-step row can vary in length
        # due to vortex lumping, but should not exceed strategy.max_vortices.
        for t in range(self.num_time_steps):
            _ = t
            # Placeholder rows: in a full implementation these rows contain
            # gamma values associated with 2D point positions.
            bound_row = np.empty((0, 2), dtype=float)
            wake_row = np.empty((0, 2), dtype=float)

            self.bound_gamma_history.append(bound_row)
            if self.strategy is None:
                wake_row_after_lumping = wake_row
            else:
                lumped_wake_rows = self.strategy.apply_lumping(self.airfoil, [wake_row])
                wake_row_after_lumping = (
                    lumped_wake_rows[0] if lumped_wake_rows else wake_row
                )
            self.wake_gamma_history.append(wake_row_after_lumping)

            self.lift_history.append(0.0)

        return SolveResult(
            bound_gamma_history=self.bound_gamma_history,
            wake_gamma_history=self.wake_gamma_history,
            lift_history=self.lift_history,
        )
