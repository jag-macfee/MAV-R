from abc import ABC, abstractmethod
from typing import List
import numpy as np

from src.airfoil import Airfoil


class VortexLumpingStrategy(ABC):
    """
    Parent class/interface for vortex lumping strategies.
    """

    @abstractmethod
    def apply_lumping(self, airfoil: Airfoil, gamma: List[np.ndarray]) -> List[np.ndarray]:
        """
        Takes in an airfoil and a circulation distribution, then applies a lumping strategy to it.

        :param airfoil: The current Airfoil instance.
        :param gamma: Time-history or current circulation distribution of vortices.
        :return: Modified gamma after lumping scheme is applied.
        """
        pass


class WCStrategy(VortexLumpingStrategy):
    """
    Represents the weighted-centroid vortex lumping strategy.
    """

    def __init__(
        self,
        max_vortices: int,
        min_lumping_distance_from_af: float,
        max_vortices_to_lump: int,
        max_relump_iterations: int
    ):
        """
        :param max_vortices: Maximum number of vortices allowed at one time.
        :param min_lumping_distance_from_af: Min distance from trailing edge before lumping begins.
        :param max_vortices_to_lump: Max number of vortices that can be lumped in a single step.
        :param max_relump_iterations: Max number of times a vortex can be considered for lumping.
        """
        self.max_vortices = max_vortices
        self.min_lumping_distance_from_af = min_lumping_distance_from_af
        self.max_vortices_to_lump = max_vortices_to_lump
        self.max_relump_iterations = max_relump_iterations

    def apply_lumping(self, airfoil: Airfoil, gamma: List[np.ndarray]) -> List[np.ndarray]:
        """
        Takes in an airfoil and a circulation distribution, applying the weighted-centroid
        lumping strategy if constraints are triggered.
        """
        gamma_after_lumping = gamma.copy()
        
        # Stub implementation
        
        return gamma_after_lumping
