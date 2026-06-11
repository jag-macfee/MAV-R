from typing import List, Optional, Callable, Tuple
import numpy as np

from src.airfoil import Airfoil
from src.vortex_strategy import VortexLumpingStrategy


class SolveResult:
    """
    A simple result container returned by Solver.solve(...).
    """

    def __init__(self, gamma_history: List[np.ndarray], lift_history: Optional[List[float]] = None):
        """
        :param gamma_history: List of k rows containing circulation distribution over time.
                              Structure: k x X x 2 (strength, position vector).
        :param lift_history: List of lift evaluations across time steps.
        """
        self.gamma_history = gamma_history
        self.lift_history = lift_history


class Solver:
    """
    Represents a single solver scenario. Stores all information needed to solve.
    """

    def __init__(
        self,
        airfoil: Airfoil,
        strategy: VortexLumpingStrategy,
        num_time_steps: int,
        Q_inf: Tuple[Callable[[float], float], Callable[[float], float]]
    ):
        """
        :param airfoil: The airfoil being simulated.
        :param strategy: The vortex lumping strategy to use.
        :param num_time_steps: Number of time steps to solve over.
        :param Q_inf: A 2D vector containing functions of t describing freestream/gust shape.
        """
        self.airfoil = airfoil
        self.strategy = strategy
        self.num_time_steps = num_time_steps
        self.Q_inf = Q_inf

        self.gamma_history: List[np.ndarray] = []
        self.lift_history: List[float] = []

    def solve(self, get_lift_history: bool = True) -> SolveResult:
        """
        Solves the configured scenario.

        :param get_lift_history: Whether to compute lift history alongside circulation.
        :return: SolveResult containing the computed histories.
        """
        self.gamma_history = []
        if get_lift_history:
            self.lift_history = []

        # Stub implementation of integration over self.num_time_steps 
        # and interactions with self.strategy
        
        for t in range(self.num_time_steps):
            pass # Apply steps...

        return SolveResult(
            gamma_history=self.gamma_history,
            lift_history=self.lift_history if get_lift_history else None
        )
